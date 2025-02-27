import os
import cv2

DATA_DIR = './Datasets'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Collect class names through console input
class_names = []
print("\nEnter class names (press Enter without typing to finish):")
while True:
    name = input(f"Class {len(class_names)+1} name: ").strip()
    if not name:
        if len(class_names) == 0:
            print("Please enter at least one class!")
            continue
        else:
            break
    class_names.append(name)

cap = cv2.VideoCapture(0)

try:
    for class_name in class_names:
        class_dir = os.path.join(DATA_DIR, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        for part in [1, 2, 3]:
            part_dir = os.path.join(class_dir, f'Part{part}')
            os.makedirs(part_dir, exist_ok=True)
            
            # Wait for 'S' to start capturing
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                cv2.putText(frame, f'Capturing {class_name}: Part {part}', (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, 'Press "S" to start | "Q" to quit', (50, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow('frame', frame)
                key = cv2.waitKey(25)
                if key == ord('s'):
                    break
                elif key == ord('q'):
                    raise KeyboardInterrupt

            # 5-second countdown
            for i in range(5, 0, -1):
                ret, frame = cap.read()
                cv2.putText(frame, str(i), (300, 300), 
                           cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 10)
                cv2.imshow('frame', frame)
                cv2.waitKey(1000)
            
            # Capture 100 images
            counter = 0
            while counter < 100:
                ret, frame = cap.read()
                cv2.imshow('frame', frame)
                cv2.waitKey(25)
                cv2.imwrite(os.path.join(part_dir, f'{counter}.jpg'), frame)
                counter += 1

            # Prompt to continue after each part
            if not (class_name == class_names[-1] and part == 3):
                while True:
                    ret, frame = cap.read()
                    cv2.putText(frame, 'Part completed!', (50, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, 'Press "S" to continue | "Q" to quit', (50, 100), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imshow('frame', frame)
                    key = cv2.waitKey(25)
                    if key == ord('s'):
                        break
                    elif key == ord('q'):
                        raise KeyboardInterrupt

except KeyboardInterrupt:
    print("\nOperation cancelled by user!")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Camera resources released.")