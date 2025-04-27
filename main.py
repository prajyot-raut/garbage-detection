import cv2
import math
import cvzone
from ultralytics import YOLO

# Path to the video file and model weights
video_path = "./test/final.mp4"
model_path = "./model.pt"       #Yolov8s with custom weights

# Define ALL class names output by your custom model in the correct order
classNames = [
    "Aerosols",
    "Aluminum can",
    "Aluminum caps",
    "Cardboard",
    "Cellulose",
    "Ceramic",
    "Combined plastic",
    "Container for household chemicals",
    "Disposable tableware",
    "Electronics",
    "Foil",
    "Furniture",
    "Glass bottle",
    "Iron utensils",
    "Liquid",
    "Metal shavings",
    "Milk bottle",
    "Organic",
    "Paper bag",
    "Paper cups",
    "Paper shavings",
    "Paper",
    "Papier mache",
    "Plastic bag",
    "Plastic bottle",
    "Plastic can",
    "Plastic canister",
    "Plastic caps",
    "Plastic cup",
    "Plastic shaker",
    "Plastic shavings",
    "Plastic toys",
    "Postal packaging",
    "Printing industry",
    "Scrap metal",
    "Stretch film",
    "Tetra pack",
    "Textile",
    "Tin",
    "Unknown plastic",
    "Wood",
    "Zip plastic bag",
    "'0'",
    "c",
    "garbage",
    "garbage_bag",
    "sampah-detection",
    "trash"
]

# Define the class names that should be considered 'garbage' for collection detection
GARBAGE_CLASS_NAMES = {
    # Generic garbage classes
    "garbage", "garbage_bag", "sampah-detection", "trash",
    
    # Specific material types
    "Aerosols", "Aluminum can", "Aluminum caps", "Cardboard", "Cellulose", 
    "Ceramic", "Combined plastic", "Container for household chemicals",
    "Disposable tableware", "Electronics", "Foil", "Furniture", "Glass bottle",
    "Iron utensils", "Liquid", "Metal shavings", "Milk bottle", "Organic",
    "Paper bag", "Paper cups", "Paper shavings", "Paper", "Papier mache",
    "Plastic bag", "Plastic bottle", "Plastic can", "Plastic canister",
    "Plastic caps", "Plastic cup", "Plastic shaker", "Plastic shavings",
    "Plastic toys", "Postal packaging", "Printing industry", "Scrap metal",
    "Stretch film", "Tetra pack", "Textile", "Tin", "Unknown plastic",
    "Wood", "Zip plastic bag"
}

# Minimum number of garbage items detected in a frame to be considered a 'collection'
MIN_ITEMS_FOR_COLLECTION = 3
# Confidence threshold for individual detections
CONFIDENCE_THRESHOLD = 0.3

# Initialize video capture
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Could not open video file: {video_path}")
    exit()

# Get video properties for positioning text
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # Get frame height

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
                    # Draw the bounding box and label directly
                    cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(0,0,255)) # Draw red rectangle
                    cvzone.putTextRect(img, f'{current_class_name} {conf}', (max(0, x1), max(35, y1)),
                                       scale=0.8, thickness=1, colorR=(0,0,255), offset=3)
            else:
                print(f"Warning: Detected class index {cls_index} out of range for classNames list.")


    # Check if a collection is detected based on the count
    collection_detected = detected_garbage_count >= MIN_ITEMS_FOR_COLLECTION

    cv2.imshow("Garbage Collection Detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Resources released.")