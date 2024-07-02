from ultralytics import YOLO
import cv2
import numpy as np
import face_recognition
import os
import sys
from PIL import Image, ImageDraw, ImageFont

model_filename = sys.argv[1] if len(sys.argv) > 1 else "best.pt"
model = YOLO(model_filename)
print(model_filename)

# Define class names and classes to show (unchanged)
classNames = ['Excavator', 'Gloves', 'Hardhat', 'Ladder', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest',
              'Person', 'SUV', 'Safety Cone', 'Safety Vest', 'bus', 'dump truck', 'fire hydrant', 'machinery',
              'mini-van', 'sedan', 'semi', 'trailer', 'truck and trailer', 'truck', 'van', 'vehicle', 'wheel loader']
classes_to_show = {
    'Hardhat': (0, 255, 0),      # Green
    'NO-Hardhat': (0, 0, 255),   # Red
    'NO-Safety Vest': (255, 0, 0),  # Blue
    'Safety Vest': (0, 255, 255),  # Yellow
    'Person': (255, 165, 0)      # Orange
}

# Load face recognition (unchanged)
reference_image_dir = r"C:\Users\h9511\Desktop\capstone\EA\Face_Recognition"
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

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

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
                    percentage_match = (1 - best_match_distance) * 100

            names.append(best_match_folder)
            percentage_matches.append(percentage_match)

    # YOLO detection
    results = model(frame)
    
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

    person_boxes = []
    safety_equipment = {}

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            
            if currentClass in classes_to_show:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                color = classes_to_show[currentClass]
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                
                conf = float(box.conf[0])
                label = f'{currentClass} {conf:.2f}'
                draw.text((x1, y1 - 10), label, fill=color)

                if currentClass == 'Person':
                    person_boxes.append((x1, y1, x2, y2))
                elif currentClass in ['Safety Vest', 'NO-Safety Vest', 'Hardhat', 'NO-Hardhat']:
                    for px1, py1, px2, py2 in person_boxes:
                        if px1 < x1 < px2 and py1 < y1 < py2:
                            if (px1, py1, px2, py2) not in safety_equipment:
                                safety_equipment[(px1, py1, px2, py2)] = []
                            safety_equipment[(px1, py1, px2, py2)].append(currentClass)

    # Draw circles for face recognition and check if identified faces are inside person boxes
    if num_faces > 0:
        for (top, right, bottom, left), name, percentage_match in zip(face_locations, names, percentage_matches):
            # Draw circle
            center = ((left + right) // 2, (top + bottom) // 2)
            radius = int((right - left) / 2)
            draw.ellipse([center[0]-radius, center[1]-radius, center[0]+radius, center[1]+radius], 
                         outline=(0, 0, 255), width=2)
            
            # Add label
            label = f"{name} ({percentage_match:.2f}%)"
            draw.text((left, top - 20), label, fill=(36, 255, 12))

            for person_box, equipment in safety_equipment.items():
                px1, py1, px2, py2 = person_box
                if px1 < left < px2 and py1 < top < py2:
                    print(f"Identified person {name}:")
                    if 'Hardhat' in equipment:
                        print("- Hardhat detected")
                    elif 'NO-Hardhat' in equipment:
                        print("- No hardhat detected")
                    if 'Safety Vest' in equipment:
                        print("- Safety vest detected")
                    elif 'NO-Safety Vest' in equipment:
                        print("- No safety vest detected")

    # Convert PIL Image back to OpenCV format for display
    frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # Display the frame
    cv2.imshow("Detection Results", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
