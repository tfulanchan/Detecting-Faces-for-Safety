# Face Recognition and Object Detection for Safety Concern

Create a folder, which is the main directory of the script

Download the model required at [here](https://github.com/rahilmoosavi/DetectConstructionSafety/blob/master/best.pt) , place it inside the main directory

Create another folder named "Face_Recognition" inside the directory of the script and folders for each identities/persons. The name of each folder should correspond to the name of each identity.

Replace the ``reference_image_dir`` with the corresponding absolute path in your directory

Create a Conda enviornment with Python version 3.9

Activate the Conda enviornment

Run script.py

``
python script.py
``
## Contributions of script-20240702-afternoon.py

- Integrated Face Recognition with Object Detection
  - Combines YOLO object detection with face recognition to identify individuals and their safety equipment.
  - Associates detected safety equipment with specific identified persons.

- Selective Safety Equipment Detection
  - Separately detects 'safety vest', 'no-safety vest', 'hardhat', and 'no-hardhat'.
  - Ensures these detections are associated with both unkown and identified persons inside 'person' rectangles.

- Person-Specific Safety Equipment Association
  - Links safety equipment detections to specific person boxes, allowing for individual-level safety compliance monitoring.

- Intelligent Reporting
  - Print statements are generated only for identified persons with associated safety equipment detections.
  - Clearly differentiates between hardhat/no-hardhat and safety vest/no-safety vest detections in reports.
  - Avoids unnecessary alerts by not printing statements for those without detected safety equipment.

- Visual Feedback
  - Draws bounding boxes for detected persons and safety equipment.
  - Adds circular annotations around recognized faces with identity labels.

- Flexible Face Recognition
  - Uses a reference image directory for face recognition, allowing easy addition or removal of known individuals.

- Real-time Processing
  - Performs all detections and recognitions in real-time on video feed.
  - Display time and local timezone of device in video feed and print statements.

