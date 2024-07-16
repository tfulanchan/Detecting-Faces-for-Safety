from ultralytics import YOLO
import cv2
import numpy as np
import face_recognition
import os
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
import tzlocal
import json
import sys

model_filename = sys.argv[1] if len(sys.argv) > 1 else "best.pt"
model = YOLO(model_filename)
print(f"Using model: {model_filename}")

# Define class names and classes to show (unchanged)
classNames = ['Excavator', 'Gloves', 'Hardhat', 'Ladder', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest',
              'Person', 'SUV', 'Safety Cone', 'Safety Vest', 'bus', 'dump truck', 'fire hydrant', 'machinery',
              'mini-van', 'sedan', 'semi', 'trailer', 'truck and trailer', 'truck', 'van', 'vehicle', 'wheel loader']

classes_to_show = {
    label: color
    for labels, color in [
        (['Hardhat', 'Safety Vest'], (0, 255, 0)),                     # Green
        # (['NO-Hardhat', 'NO-Safety Vest'], (255, 192, 203)),  # Pink
        (['NO-Hardhat', 'NO-Safety Vest'], (255, 0, 0)),  # Pink
        # (['Safety Vest'], (0, 255, 255)),               # Yellow
        (['Person'], (255, 165, 0))                     # Orange
    ]
    for label in labels
}

# Load face recognition (unchanged)
reference_image_dir = "Face_Recognition"
reference_data = {}
for folder_name in os.listdir(reference_image_dir):
    folder_path = os.path.join(reference_image_dir, folder_name)
    if os.path.isdir(folder_path):
        folder_encodings = []
        for filename in os.listdir(folder_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(folder_path, filename)
                reference_image = face_recognition.load_image_file(image_path)
                reference_encoding = face_recognition.face_encodings(reference_image)[0]
                folder_encodings.append(reference_encoding)
        reference_data[folder_name] = folder_encodings

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Assuming reference_image_dir is a subdirectory of the script directory
reference_image_dir = os.path.join(script_dir, "Face_Recognition")

# Load ACL from acl.json in reference_image_dir
acl_file_path = os.path.join(reference_image_dir, "acl.json")

# Print the path for debugging
print(f"Attempting to open ACL file at: {acl_file_path}")

try:
    with open(acl_file_path, 'r') as acl_file:
        acl = json.load(acl_file)
except FileNotFoundError:
    acl = {}  # Create an empty ACL if file is not found
except json.JSONDecodeError:
    print(f"Error decoding JSON from {acl_file_path}")
    acl = {}  # Create an empty ACL if JSON is invalid

# Get the local timezone
local_timezone = tzlocal.get_localzone()

def process_frame(frame, time_str):
    # Face recognition
    face_locations = face_recognition.face_locations(frame)
    num_faces = len(face_locations)
    
    names = []
    percentage_matches = []

    if num_faces > 0:
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        for face_encoding in face_encodings:
            best_match_folder = "Unknown"
            best_match_distance = 0.6
            percentage_match = 0.0

            for folder_name, folder_encodings in reference_data.items():
                face_distances = face_recognition.face_distance(np.array(folder_encodings), face_encoding)
                average_distance = np.mean(face_distances)

                if average_distance < best_match_distance:
                    best_match_folder = folder_name
                    best_match_distance = average_distance
                    percentage_match = (1 - best_match_distance)

            names.append(best_match_folder)
            percentage_matches.append(percentage_match)

    # YOLO detection
    results = model(frame)
    
    # Convert the frame to a PIL Image
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

    # Add timestamp to the image
    font = ImageFont.truetype("arial.ttf", 20)  # Adjust font and size as needed
    draw.text((10, 10), time_str, font=font, fill=(255, 255, 255))

    person_boxes = []
    safety_equipment = {}

    margin = 10  # Margin for expanding person bounding box

    # First pass: Detect persons and create person boxes (without drawing)
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            
            if currentClass == 'Person':
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                # Expand the bounding box by the margin
                x1, y1, x2, y2 = x1 - margin, y1 - margin, x2 + margin, y2 + margin
                person_boxes.append((x1, y1, x2, y2))
                safety_equipment[(x1, y1, x2, y2)] = {"Hardhat": False, "Safety Vest": False, "No Hardhat": False, "No Safety Vest": False}

    # Second pass: Detect safety equipment and associate with persons
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            
            if currentClass in classes_to_show and currentClass != 'Person':
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                color = classes_to_show[currentClass]
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                
                conf = float(box.conf[0])
                label = f'{currentClass} {conf:.2f}'
                bbox = draw.textbbox((x1, y1), label, font=font)
                draw.rectangle(bbox, fill=color)
                draw.text((x1, bbox[1]), label, font=font, fill=(0, 0, 0))

                if currentClass in ['Safety Vest', 'NO-Safety Vest', 'Hardhat', 'NO-Hardhat']:
                    for px1, py1, px2, py2 in person_boxes:
                        if px1 < x1 < px2 and py1 < y1 < py2:
                            if currentClass == 'NO-Hardhat':
                                safety_equipment[(px1, py1, px2, py2)]["No Hardhat"] = True
                            elif currentClass == 'NO-Safety Vest':
                                safety_equipment[(px1, py1, px2, py2)]["No Safety Vest"] = True
                            else:
                                safety_equipment[(px1, py1, px2, py2)][currentClass] = True

    # Resolve conflicts for each person
    for person_box in person_boxes:
        if safety_equipment[person_box]["Safety Vest"]:
            safety_equipment[person_box]["No Safety Vest"] = False
        if safety_equipment[person_box]["Hardhat"]:
            safety_equipment[person_box]["No Hardhat"] = False

    # Process detected persons and their safety equipment
    print(f"\n--- Detection Results at {time_str} ---")
    for i, (person_box, equipment) in enumerate(safety_equipment.items()):
        px1, py1, px2, py2 = person_box
        person_name = "Unknown"
        is_authorized = False
        
        # Check if this person box contains an identified face
        for (top, right, bottom, left), name, percentage_match in zip(face_locations, names, percentage_matches):
            if px1 < left < px2 and py1 < top < py2:
                person_name = name
                # Check if the person is authorized
                if name != "Unknown" and (name not in acl or acl[name] != "unauthorized"):
                    is_authorized = True
                
                # Draw circle for identified face
                center = ((left + right) // 2, (top + bottom) // 2)
                radius = int((right - left) / 2)
                circle_color = (255, 255, 0) if is_authorized else (255, 0, 0)  # Yellow if authorized, Red if not
                draw.ellipse([center[0]-radius, center[1]-radius, center[0]+radius, center[1]+radius], 
                             outline=circle_color, width=2)
                
                # Draw rectangle for unauthorized faces
                if not is_authorized:
                    draw.rectangle([left, top, right, bottom], outline=(255, 0, 0), width=2)
                
                # Add label for identified face
                label = f"{name} {percentage_match:.2f}"
                bbox = draw.textbbox((left, top), label, font=font)
                draw.rectangle(bbox, fill=circle_color)
                draw.text((left, bbox[1]), label, font=font, fill=(0, 0, 0))
                break

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

    # Convert PIL Image back to OpenCV format for display
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def process_image(image_path):
    frame = cv2.imread(image_path)
    current_time = datetime.now(local_timezone)
    time_str = current_time.strftime("%Y-%m-%d %H:%M:%S %Z")
    processed_frame = process_frame(frame, time_str)
    cv2.imshow("Local-host Face Recognition + YOLO", processed_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = datetime.now(local_timezone)
        time_str = current_time.strftime("%Y-%m-%d %H:%M:%S %Z")
        processed_frame = process_frame(frame, time_str)
        cv2.imshow("Local-host Face Recognition + YOLO", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

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
