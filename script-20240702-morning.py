from ultralytics import YOLO
import cv2
import numpy as np
from ultralytics import YOLO
import cv2
import face_recognition
import numpy as np
import os
import sys
from PIL import Image, ImageDraw, ImageFont

model_filename = sys.argv[1] if len(sys.argv) > 1 else "best.pt"
model = YOLO(model_filename)
print(model_filename)
# Define class names
classNames = ['Excavator', 'Gloves', 'Hardhat', 'Ladder', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest',
              'Person', 'SUV', 'Safety Cone', 'Safety Vest', 'bus', 'dump truck', 'fire hydrant', 'machinery',
              'mini-van', 'sedan', 'semi', 'trailer', 'truck and trailer', 'truck', 'van', 'vehicle', 'wheel loader']
# Define the classes to display and their colors
classes_to_show = {
    'Hardhat': (0, 255, 0),      # Green
    'NO-Hardhat': (0, 0, 255),   # Red
    'NO-Safety Vest': (255, 0, 0),  # Blue
    'Safety Vest': (0, 255, 255)  # Yellow
}

# Load face recognition
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
    # inference on frame face recognition
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)
    if len(face_locations) > 0:
        names = []
        percentage_matches = []

        for face_encoding in face_encodings:
            if len(reference_data) > 0:
                best_match_folder = "Unknown"
                best_match_distance = 0.6
                percentage_match = None  # Initialize with a default value

                for folder_name, folder_encodings in reference_data.items():
                    face_distances = face_recognition.face_distance(np.array(folder_encodings), face_encoding)
                    average_distance = np.mean(face_distances)

                    if average_distance < best_match_distance:
                        best_match_folder = folder_name
                        best_match_distance = average_distance

                if best_match_folder != "Unknown":
                    percentage_match = (1 - best_match_distance) * 100
                else:
                    percentage_match = 0.0  # Assign a value for "Unknown" case

            else:
                best_match_folder = "Unknown"
                percentage_match = 0.0  # Assign a value for the case when reference_data is empty

            names.append(best_match_folder)
            percentage_matches.append(percentage_match)

        # rectangle
        # for (top, right, bottom, left), name, percentage_match in zip(face_locations, names, percentage_matches):
        #     cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        #     if percentage_match > 0.0:
        #         cv2.putText(frame, f"{name} ({percentage_match:.2f}%)", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)
        #     else:
        #         cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)

        # elipse
        for (top, right, bottom, left), name, percentage_match in zip(face_locations, names, percentage_matches):
            center = ((left + right) // 2, (top + bottom) // 2)
            axes = ((right - left) // 2, (bottom - top) // 2)
            cv2.ellipse(frame, center, axes, 0, 0, 360, (0, 0, 255), 2)
            
            if percentage_match > 0.0:
                text = f"{name} ({percentage_match:.2f}%)"
            else:
                text = name
            
            cv2.putText(frame, text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)

    # Perform inference on the frame
    results = model(frame)
    # Convert the frame to a PIL Image
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    # Process the results and draw only the desired classes
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            # Check if the detected class is in our list of classes to show
            if currentClass in classes_to_show:
                # Get the bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # Get color for the current class
                color = classes_to_show[currentClass]
                # Draw the rectangle
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                # Add label
                conf = float(box.conf[0])
                label = f'{currentClass} {conf:.2f}'
                draw.text((x1, y1 - 10), label, fill=color)
    # Convert the PIL Image back to a numpy array for display with OpenCV
    frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # Display the frame
    cv2.imshow("Detection Results", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
