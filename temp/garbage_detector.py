import cv2
from ultralytics import YOLO

custom_model_path = './best.pt' 
# Replace with the path to your CCTV footage file or use 0 for webcam
video_path = './garbage_5.jpeg'
# Set confidence threshold (only detections above this threshold will be shown)
confidence_threshold = 0.5
# ===> 2. Update with the class IDs YOUR custom model uses for garbage <===
# Check the training output or configuration (e.g., data.yaml) for your custom model
# If your model only detects one type of garbage, it might just be {0}
# If it detects multiple types, list all relevant IDs, e.g., {0, 1, 2}
GARBAGE_CLASS_IDS = {0} # <-- IMPORTANT: Update this based on your custom model!
# --- End Configuration ---


# Load YOUR custom YOLO model
try:
    model = YOLO(custom_model_path)
    print(f"Loaded custom model from: {custom_model_path}")
except Exception as e:
    print(f"Error loading custom model: {e}")
    print("Please ensure the 'custom_model_path' is correct.")
    exit()

try:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        exit()

    while True:
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            garbage_detected = False
            # Run YOLOv8 inference on the frame
            results = model(frame, conf=confidence_threshold, verbose=False)

            # Check if any detected object belongs to the garbage classes
            for box in results[0].boxes:
                cls_id = int(box.cls[0].item())
                if cls_id in GARBAGE_CLASS_IDS:
                    garbage_detected = True
                    break # Found garbage, no need to check further in this frame

            # Display status text on the frame
            status_text = "Garbage Detected" if garbage_detected else "No Garbage Detected"
            color = (0, 0, 255) if garbage_detected else (0, 255, 0) # Red if detected, Green if not

            if garbage_detected:
                print("Garbage detected in the frame!")
            
            # Put text on the top-left corner
            cv2.putText(frame, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, color, 2, cv2.LINE_AA)

            # Display the frame with status text
            cv2.imshow("Garbage Detection Status", frame) # Use the frame directly, changed window title

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            print("End of video reached or error reading frame.")
            break

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Release the video capture object and close the display window
    if 'cap' in locals() and cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
    print("Resources released.")

