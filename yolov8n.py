import cv2
import time
from collections import Counter
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('Custom_Model.pt')

# Open the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize variables
fps = 2  # Webcam FPS
frame_interval = 1 / fps  # Time per frame
detected_signs = []  # List to store detected gestures
gesture_buffer = []  # Temporary buffer for gestures detected in 1 second
start_time = time.time()  # Track time for gesture analysis

# Define constants
MAX_SIGNS = 5  # Maximum signs in the list
DISPLAY_TIME = 2  # Time to display the list (in seconds)
ANALYSIS_DURATION = 1  # Duration for analyzing gestures (in seconds)

print("Press 'E' to display detected signs. Press 'Q' to quit.")

while True:
    frame_start_time = time.time()  # Start time for the current frame

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Perform inference
    results = model(frame)
    current_gestures = []

    # Extract detection data
    for result in results:
        classes = result.boxes.cls.cpu().numpy()  # Class IDs
        for cls in classes:
            gesture_name = model.names[int(cls)]
            current_gestures.append(gesture_name)

    # Add detected gestures to the buffer
    gesture_buffer.extend(current_gestures)

    # Annotate the frame with bounding boxes and labels
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
        classes = result.boxes.cls.cpu().numpy()  # Class IDs

        for box, cls in zip(boxes, classes):
            x1, y1, x2, y2 = map(int, box)
            label = f"{model.names[int(cls)]}"
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the annotated frame
    cv2.imshow('Sign Language Detection', frame)

    # Check if ANALYSIS_DURATION has passed
    if time.time() - start_time >= ANALYSIS_DURATION:
        if gesture_buffer:
            # Find the most frequent gesture in the buffer
            most_common_gesture = Counter(gesture_buffer).most_common(1)[0][0]

            # Add the gesture to the list if it's not already at the end
            if len(detected_signs) == 0 or detected_signs[-1] != most_common_gesture:
                detected_signs.append(most_common_gesture)

            # Keep the list size within the maximum limit
            if len(detected_signs) > MAX_SIGNS:
                detected_signs.pop(0)

        # Clear the buffer and reset the timer
        gesture_buffer = []
        start_time = time.time()

    # Check for user input
    key = cv2.waitKey(1) & 0xFF

    if key == ord('e'):  # Display detected signs
        # Show the detected signs on the screen
        display_frame = frame.copy()
        display_text = "Detected Signs: " + ", ".join(detected_signs)
        cv2.putText(display_frame, display_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Detected Signs', display_frame)
        cv2.waitKey(DISPLAY_TIME * 1000)  # Wait for DISPLAY_TIME seconds
    elif key == ord('q'):  # Quit the program
        break

    # Enforce 1 FPS (delay the next iteration)
    elapsed_time = time.time() - frame_start_time
    if elapsed_time < frame_interval:
        time.sleep(frame_interval - elapsed_time)

# Release resources
cap.release()
cv2.destroyAllWindows()
