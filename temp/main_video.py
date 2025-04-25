import cv2
import math
import cvzone
from ultralytics import YOLO

# --- Configuration ---
video_path = "./final.mp4"
model_path = "./best.pt"
# Define ALL class names output by your custom model in the correct order
classNames = ['0', 'c', 'garbage', 'garbage_bag', 'sampah-detection', 'trash'] 
# Define the class names that should be considered 'garbage' for collection detection
GARBAGE_CLASS_NAMES = {'garbage', 'garbage_bag', 'sampah-detection', 'trash'} 
# Minimum number of garbage items detected in a frame to be considered a 'collection'
MIN_ITEMS_FOR_COLLECTION = 3 
# Confidence threshold for individual detections
CONFIDENCE_THRESHOLD = 0.3 # Adjusted confidence slightly lower to potentially catch more items in a pile
# --- End Configuration ---

# Initialize video capture
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Could not open video file: {video_path}")
    exit()

# Load YOLO model with custom weights
try:
    model = YOLO(model_path)
    print(f"Loaded custom model from: {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()


while True:
    success, img = cap.read()
    if not success:
        print("End of video or error reading frame.")
        break

    results = model(img, stream=True, conf=CONFIDENCE_THRESHOLD, verbose=False)
    
    detected_garbage_count = 0
    garbage_boxes_for_drawing = [] # Store boxes to draw only if collection is detected

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Confidence and Class
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls_index = int(box.cls[0])
            
            # Ensure class index is valid
            if cls_index < len(classNames):
                current_class_name = classNames[cls_index]

                # Check if the detected class is considered garbage
                if current_class_name in GARBAGE_CLASS_NAMES:
                    detected_garbage_count += 1
                    # Store info for potential drawing later
                    garbage_boxes_for_drawing.append({
                        'bbox': (x1, y1, w, h),
                        'class': current_class_name,
                        'conf': conf
                    })
            else:
                print(f"Warning: Detected class index {cls_index} out of range for classNames list.")


    # Check if a collection is detected based on the count
    collection_detected = detected_garbage_count >= MIN_ITEMS_FOR_COLLECTION

    if collection_detected:
        # Option 1: Display text indicating collection detected
        cvzone.putTextRect(img, f'Garbage Collection Detected ({detected_garbage_count} items)', 
                           (50, 50), scale=1.5, thickness=2, colorR=(0,0,255))
        
        # Option 2: Draw boxes for items part of the collection (uncomment if desired)
        # for item in garbage_boxes_for_drawing:
        #     x1, y1, w, h = item['bbox']
        #     label = f"{item['class']} {item['conf']}"
        #     cvzone.cornerRect(img, (x1, y1, w, h), t=2, colorC=(0, 0, 255)) # Red box for collection items
        #     cvzone.putTextRect(img, label, (max(0, x1), max(35, y1)), scale=1, thickness=1)

    # If you want to explicitly show "No Collection" when few items are detected:
    # else:
    #     cvzone.putTextRect(img, 'No Collection Detected', (50, 50), scale=1.5, thickness=2, colorR=(0,255,0))


    cv2.imshow("Garbage Collection Detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Resources released.")