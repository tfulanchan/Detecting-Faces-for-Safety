import cv2
import numpy as np
import asyncio
import io
import os
import sys
import time
import uuid
from PIL import Image, ImageDraw, ImageFont
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.face.models import TrainingStatusType, Person, QualityForRecognition
import requests
import json
# Azure Face API credentials
FACE_KEY = "FACE_KEY"
FACE_ENDPOINT = "FACE_ENDPOINT"
from datetime import datetime
import tzlocal
# Azure Custom Vision credentials
CUSTOM_VISION_KEY = "CUSTOM_VISION_KEY"
CUSTOM_VISION_ENDPOINT = "CUSTOM_VISION_ENDPOINT"

# Create an authenticated FaceClient
face_client = FaceClient(FACE_ENDPOINT, CognitiveServicesCredentials(FACE_KEY))

# Create the PersonGroup
PERSON_GROUP_ID = str(uuid.uuid4())  # assign a random ID
print('Person group:', PERSON_GROUP_ID)
face_client.person_group.create(person_group_id=PERSON_GROUP_ID, name=PERSON_GROUP_ID, recognition_model='recognition_04')

def is_image_valid(image_path):
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            
            if width < 36 or height < 36 or width > 4096 or height > 4096:
                return False
            
            if os.path.getsize(image_path) > 6 * 1024 * 1024:
                return False
            
            return True
    except Exception:
        return False

def add_faces_to_person(person_name, person_id, image_paths):
    for image_path in image_paths:
        if is_image_valid(image_path):
            try:
                with open(image_path, 'rb') as image_file:
                    face_client.person_group_person.add_face_from_stream(
                        PERSON_GROUP_ID, 
                        person_id, 
                        image_file,
                        detection_model='detection_03',
                        recognition_model='recognition_04'
                    )
            except Exception:
                pass  # Silently ignore errors

# Path to your 'faces' folder
faces_folder = 'faces'

# Iterate through subfolders in the 'faces' folder
for person_name in os.listdir(faces_folder):
    person_folder = os.path.join(faces_folder, person_name)
    if os.path.isdir(person_folder):
        try:
            person = face_client.person_group_person.create(PERSON_GROUP_ID, name=person_name)
            
            image_paths = [os.path.join(person_folder, f) for f in os.listdir(person_folder) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            faces_added = add_faces_to_person(person_name, person.person_id, image_paths)
            
            if faces_added == 0:
                # print(f"No valid faces were added for {person_name}. Deleting person from group.")
                face_client.person_group_person.delete(PERSON_GROUP_ID, person.person_id)
            # else:
                # print(f"Added {faces_added} faces for {person_name}")
        except Exception as e:
            print(f"Error processing person {person_name}: {str(e)}")

# Train PersonGroup
print("Training the person group...")
face_client.person_group.train(PERSON_GROUP_ID)
while True:
    training_status = face_client.person_group.get_training_status(PERSON_GROUP_ID)
    print("Training status: {}.".format(training_status.status))
    if training_status.status is TrainingStatusType.succeeded:
        break
    elif training_status.status is TrainingStatusType.failed:
        print(f"Training failed. Error message: {training_status.message}")
        face_client.person_group.delete(person_group_id=PERSON_GROUP_ID)
        sys.exit('Training the person group has failed.')
    time.sleep(5)

def predict_custom_vision(image_data):
    headers = {
        'Prediction-Key': CUSTOM_VISION_KEY,
        'Content-Type': 'application/octet-stream'
    }
    
    response = requests.post(CUSTOM_VISION_ENDPOINT, headers=headers, data=image_data)
    if response.status_code == 200:
        results = response.json()
        return results['predictions']
    else:
        print(f"Error in Custom Vision API call: {response.status_code} - {response.text}")
        return []

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Path to the 'faces' folder
faces_dir = os.path.join(script_dir, 'faces')

# Load ACL
acl_file_path = os.path.join(faces_dir, "acl.json")
with open(acl_file_path, 'r') as acl_file:
    acl = json.load(acl_file)

# Load list of authorized persons (folders in 'faces' directory)
authorized_persons = set(os.listdir(faces_dir))
# Remove 'acl.json' from the set of authorized persons if it's there
authorized_persons.discard('acl.json')
from datetime import datetime

def get_current_time():
    local_tz = tzlocal.get_localzone()  # Replace with your local timezone
    current_time = datetime.now(local_tz)
    return current_time.strftime("%Y-%m-%d %H:%M:%S %Z")

def process_frame(frame, time_str=None):
    if time_str is None:
        time_str = get_current_time()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb_frame)
    draw = ImageDraw.Draw(img)
    
    font = ImageFont.truetype("arial.ttf", 20)
    
    # Add timestamp to the image
    draw.text((10, 10), time_str, font=font, fill=(255, 255, 255))

    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()

    person_boxes = []
    safety_equipment = {}
    margin = 10  # Margin for expanding person bounding box

    try:
        detected_faces = face_client.face.detect_with_stream(io.BytesIO(img_byte_arr), detection_model='detection_03', recognition_model='recognition_04')
        
        face_ids = [face.face_id for face in detected_faces]
        
        if face_ids:
            results = face_client.face.identify(face_ids, PERSON_GROUP_ID)
        else:
            results = []

        custom_vision_results = predict_custom_vision(img_byte_arr)

        # First pass: Detect persons and create person boxes
        for face in detected_faces:
            rect = face.face_rectangle
            x1, y1, x2, y2 = rect.left - margin, rect.top - margin, rect.left + rect.width + margin, rect.top + rect.height + margin
            person_boxes.append((x1, y1, x2, y2))
            safety_equipment[(x1, y1, x2, y2)] = {"Hardhat": False, "Safety Vest": False, "No Hardhat": False, "No Safety Vest": False}

        # Second pass: Detect safety equipment and associate with persons
        for prediction in custom_vision_results:
            if prediction['probability'] > 0.5:
                box = prediction['boundingBox']
                left = int(box['left'] * img.width)
                top = int(box['top'] * img.height)
                width = int(box['width'] * img.width)
                height = int(box['height'] * img.height)
                
                if prediction['tagName'].lower() in ['hardhat', 'safetyvest']:
                    color = (0, 255, 0)  # Green
                elif prediction['tagName'].lower() in ['no-hardhat', 'no-safetyvest']:
                    color = (255, 0, 0)  # Pink
                else:
                    color = (0, 255, 255)  # Cyan (default)
                
                draw.rectangle([left, top, left + width, top + height], outline=color, width=2)
                label = f"{prediction['tagName']} {prediction['probability']:.2f}"
                bbox = draw.textbbox((left, top), label, font=font)
                draw.rectangle(bbox, fill=color)
                draw.text((left, top), label, font=font, fill=(0, 0, 0))

                for px1, py1, px2, py2 in person_boxes:
                    if px1 < left < px2 and py1 < top < py2:
                        if prediction['tagName'].lower() == 'no-hardhat':
                            safety_equipment[(px1, py1, px2, py2)]["No Hardhat"] = True
                        elif prediction['tagName'].lower() == 'no-safetyvest':
                            safety_equipment[(px1, py1, px2, py2)]["No Safety Vest"] = True
                        elif prediction['tagName'].lower() == 'hardhat':
                            safety_equipment[(px1, py1, px2, py2)]["Hardhat"] = True
                        elif prediction['tagName'].lower() == 'safetyvest':
                            safety_equipment[(px1, py1, px2, py2)]["Safety Vest"] = True

        # Resolve conflicts for each person
        for person_box in person_boxes:
            if safety_equipment[person_box]["Safety Vest"]:
                safety_equipment[person_box]["No Safety Vest"] = False
            if safety_equipment[person_box]["Hardhat"]:
                safety_equipment[person_box]["No Hardhat"] = False

        print(f"\n--- Detection Results at {time_str} ---")
        for i, (person_box, equipment) in enumerate(safety_equipment.items()):
            px1, py1, px2, py2 = person_box
            person_name = "Unknown"
            is_authorized = False
            confidence = 0.0

            for face, result in zip(detected_faces, results):
                rect = face.face_rectangle
                if px1 < rect.left < px2 and py1 < rect.top < py2:
                    if result.candidates:
                        person = face_client.person_group_person.get(PERSON_GROUP_ID, result.candidates[0].person_id)
                        person_name = person.name
                        confidence = result.candidates[0].confidence
                        
                        # Check authorization
                        if person_name in authorized_persons and (person_name not in acl or acl.get(person_name) != "unauthorized"):
                            is_authorized = True
                    break

            # Draw circle for face
            center = (rect.left + rect.width // 2, rect.top + rect.height // 2)
            radius = min(rect.width, rect.height) // 2
            circle_color = (255, 255, 0) if is_authorized else (255, 0, 0)  # Yellow if authorized, Red if not
            draw.ellipse([center[0]-radius, center[1]-radius, center[0]+radius, center[1]+radius], 
                        outline=circle_color, width=2)
            
            # Add label for identified face
            label = f"{person_name} {confidence:.2f}"
            bbox = draw.textbbox((rect.left, rect.top), label, font=font)
            draw.rectangle(bbox, fill=circle_color)
            draw.text((rect.left, rect.top), label, font=font, fill=(0, 0, 0))

            print(f"Person {i+1} ({person_name}):")
            print(f"- {'Authorized' if is_authorized else 'Unauthorized'}")
            if equipment["No Hardhat"]:
                print("- No hardhat detected")
            elif equipment["Hardhat"]:
                print("- Hardhat detected")

            if equipment["Safety Vest"]:
                print("- Safety vest detected")
            elif equipment["No Safety Vest"]:
                print("- No safety vest detected")
    except Exception as e:
        print(f"Error processing frame: {str(e)}")

    # Convert PIL Image back to OpenCV format
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def process_image(image_path):
    frame = cv2.imread(image_path)
    current_time = get_current_time()
    processed_frame = process_frame(frame, current_time)
    cv2.imshow('Azure Face API + Custom Vision', processed_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        current_time = get_current_time()    
        processed_frame = process_frame(frame, current_time)
        cv2.imshow('Azure Face API + Custom Vision', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Main execution
if __name__ == "__main__":
    print("Choose the type of input:")
    print("1. Image")
    print("2. Video")
    choice = input("Enter your choice (1 or 2): ")

    if choice == '1':
        image_path = input("Enter the path to the image file: ")
        process_image(image_path)
    elif choice == '2':
        video_path = input("Enter the path to the video file: ")
        process_video(video_path)
    else:
        print("Invalid choice. Please run the script again and choose 1 or 2.")
