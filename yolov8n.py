
import cv2
import time
from collections import Counter
from ultralytics import YOLO

# Constants
FPS = 2
MAX_SIGNS = 5
DISPLAY_TIME = 2
ANALYSIS_DURATION = 1
MODEL_PATH = 'Custom_Model.pt'

def initialize_model_and_camera():
    """Initialize YOLO model and webcam"""
    model = YOLO(MODEL_PATH)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Error: Could not open webcam.")
    
    return model, cap

def process_frame(model, frame):
    """Process a frame and return detection results"""
    return model(frame)

def draw_annotations(frame, results, model):
    """Draw bounding boxes and labels on the frame"""
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        for box, cls in zip(boxes, classes):
            x1, y1, x2, y2 = map(int, box)
            label = f"{model.names[int(cls)]}"
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame

def update_gesture_list(gesture_buffer, detected_signs, analysis_duration, max_signs, start_time):
    """Update the detected signs list based on gesture buffer"""
    if time.time() - start_time >= analysis_duration:
        if gesture_buffer:
            most_common = Counter(gesture_buffer).most_common(1)[0][0]
            if not detected_signs or detected_signs[-1] != most_common:
                detected_signs.append(most_common)
                detected_signs = detected_signs[-max_signs:]
        return [], time.time()  # Reset buffer and update start_time
    return gesture_buffer, start_time  # Return unchanged


def handle_user_input(frame, detected_signs, display_time):
    """Handle keyboard input and display detected signs"""
    key = cv2.waitKey(1) & 0xFF
    if key == ord('e'):
        display_frame = frame.copy()
        display_text = "Detected Signs: " + ", ".join(detected_signs)
        cv2.putText(display_frame, display_text, (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Detected Signs', display_frame)
        cv2.waitKey(display_time * 1000)
    return key

def enforce_fps(frame_start_time, fps):
    """Maintain specified FPS"""
    elapsed = time.time() - frame_start_time
    if elapsed < 1/fps:
        time.sleep((1/fps) - elapsed)

def main():
    """Main application loop"""
    model, cap = initialize_model_and_camera()
    detected_signs = []
    gesture_buffer = []
    start_time = time.time()

    print("Press 'E' to display detected signs. Press 'Q' to quit.")

    while True:
        frame_start = time.time()
        
        # Read and process frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        results = process_frame(model, frame)
        annotated_frame = draw_annotations(frame, results, model)
        cv2.imshow('Sign Language Detection', annotated_frame)

        # Update gestures (fixed argument list)
        current_gestures = [model.names[int(cls)] 
                           for result in results 
                           for cls in result.boxes.cls.cpu().numpy() 
                           if result.boxes is not None]  # Added safety check
        
        gesture_buffer.extend(current_gestures)
        gesture_buffer, start_time = update_gesture_list(
            gesture_buffer, 
            detected_signs, 
            ANALYSIS_DURATION, 
            MAX_SIGNS, 
            start_time  # Now passing start_time
        )

        # Handle input and FPS
        key = handle_user_input(annotated_frame, detected_signs, DISPLAY_TIME)  # Changed to annotated_frame
        if key == ord('q'):
            break
            
        enforce_fps(frame_start, FPS)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()