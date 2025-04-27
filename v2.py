import cv2
import math
import cvzone
from ultralytics import YOLO
import datetime # Import datetime module
import numpy as np # Import numpy
import random # Import random module

# --- Configuration ---
video_path = "./final.mp4"
model_path = "./final.pt"       #Yolov8s with custom weights

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
GARBAGE_CLASS_NAMES = {'garbage', 'garbage_bag', 'sampah-detection', 'trash'}
# Minimum number of garbage items detected in a frame to be considered a 'collection'
MIN_ITEMS_FOR_COLLECTION = 3
# Confidence threshold for individual detections
CONFIDENCE_THRESHOLD = 0.3 # Adjusted confidence slightly lower to potentially catch more items in a pile

# --- Bottom Left Popup Configuration ---
GOOGLE_LOGO_PATH = "google_logo.png" # Provide path to Google logo image
MAP_PIN_PATH = "map_pin.png"       # Provide path to map pin image
LOGO_WIDTH = 80
PIN_HEIGHT = 40
POPUP_PADDING = 10 # Padding around elements and border
POPUP_ALPHA = 0.7 # Background transparency
POPUP_TEXT_SCALE = 0.5
POPUP_TEXT_THICKNESS = 1
POPUP_TEXT_COLOR = (255, 255, 255) # White
POPUP_BG_COLOR = (0, 0, 0) # Black

# --- Coordinate Simulation ---
BASE_LAT = 30.637753
BASE_LONG = 76.724986
# Approx 3 meters offset in degrees (1 deg lat ~ 111km, 1 deg lon ~ 96km at this lat)
MAX_COORD_OFFSET = 0.000035 # ~3 meters / (avg meters per degree)
# --- End Configuration ---

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

# --- Load and Prepare Assets ---
try:
    google_logo = cv2.imread(GOOGLE_LOGO_PATH, cv2.IMREAD_UNCHANGED) # Load with alpha if available
    if google_logo is None: raise FileNotFoundError
    # Resize logo maintaining aspect ratio based on width
    logo_h, logo_w = google_logo.shape[:2]
    ratio = LOGO_WIDTH / logo_w
    logo_resized_h = int(logo_h * ratio)
    google_logo_resized = cv2.resize(google_logo, (LOGO_WIDTH, logo_resized_h))
    print(f"Loaded and resized Google logo.")
except FileNotFoundError:
    print(f"Error: Could not load Google logo from {GOOGLE_LOGO_PATH}. Skipping logo.")
    google_logo_resized = None
except Exception as e:
    print(f"Error processing Google logo: {e}. Skipping logo.")
    google_logo_resized = None

try:
    map_pin = cv2.imread(MAP_PIN_PATH, cv2.IMREAD_UNCHANGED) # Load with alpha if available
    if map_pin is None: raise FileNotFoundError
    # Resize pin maintaining aspect ratio based on height
    pin_h, pin_w = map_pin.shape[:2]
    ratio = PIN_HEIGHT / pin_h
    pin_resized_w = int(pin_w * ratio)
    map_pin_resized = cv2.resize(map_pin, (pin_resized_w, PIN_HEIGHT))
    print(f"Loaded and resized map pin.")
except FileNotFoundError:
    print(f"Error: Could not load map pin from {MAP_PIN_PATH}. Skipping pin.")
    map_pin_resized = None
except Exception as e:
    print(f"Error processing map pin: {e}. Skipping pin.")
    map_pin_resized = None

# --- Fake Location Data ---
location_line1 = "Manauli, Punjab, India"
location_line2 = "JPQF+6RW, Sector 83, JLPL Industrial"
location_line3 = "Area, Manauli, Punjab 140306, India"
# location_line4 will be generated dynamically in the loop
# --- End Fake Location Data ---


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

    # --- Add Bottom Left Popup ---
    # Get current timestamp
    now = datetime.datetime.now()
    timestamp_str = now.strftime("%d/%m/%y %I:%M:%S %p") # Format like the image

    # --- Simulate Coordinate Change ---
    current_lat = BASE_LAT + random.uniform(-MAX_COORD_OFFSET, MAX_COORD_OFFSET)
    current_long = BASE_LONG + random.uniform(-MAX_COORD_OFFSET, MAX_COORD_OFFSET)
    location_line4 = f"Lat {current_lat:.6f} Long {current_long:.6f}" # Format to 6 decimal places

    # Define text lines for the right column (including updated coordinates)
    text_lines = [
        location_line1,
        location_line2,
        location_line3,
        location_line4, # Use the updated line
        timestamp_str
    ]

    # --- Calculate Dimensions ---
    # Text block dimensions
    max_text_width = 0
    line_height = 0
    text_line_spacing = 5
    for line in text_lines:
        (w, h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, POPUP_TEXT_SCALE, POPUP_TEXT_THICKNESS)
        if w > max_text_width:
            max_text_width = w
        line_height = max(line_height, h) # Use max height found
    text_block_height = (line_height + text_line_spacing) * len(text_lines) - text_line_spacing

    # Left column dimensions (use actual resized dimensions if images loaded)
    left_col_width = 0
    left_col_height = 0
    logo_h, logo_w = (google_logo_resized.shape[:2] if google_logo_resized is not None else (0, 0))
    pin_h, pin_w = (map_pin_resized.shape[:2] if map_pin_resized is not None else (0, 0))

    if google_logo_resized is not None:
        left_col_width = max(left_col_width, logo_w)
        left_col_height += logo_h + POPUP_PADDING # Add padding between logo and pin
    if map_pin_resized is not None:
        left_col_width = max(left_col_width, pin_w)
        left_col_height += pin_h

    # Total popup dimensions
    popup_content_width = left_col_width + POPUP_PADDING + max_text_width
    popup_width = popup_content_width + 2 * POPUP_PADDING
    popup_height = max(left_col_height, text_block_height) + 2 * POPUP_PADDING

    # --- Calculate Position ---
    popup_x = POPUP_PADDING # Position from left edge
    popup_y = frame_height - popup_height - POPUP_PADDING # Position from bottom edge

    # Ensure popup stays within frame boundaries (basic check)
    if popup_y < 0: popup_y = 0
    if popup_x + popup_width > frame_width: popup_width = frame_width - popup_x
    if popup_y + popup_height > frame_height: popup_height = frame_height - popup_y


    # --- Draw Popup ---
    try:
        # Create overlay for transparency
        overlay = img.copy()

        # Draw background rectangle
        cv2.rectangle(overlay, (popup_x, popup_y), (popup_x + popup_width, popup_y + popup_height),
                      POPUP_BG_COLOR, -1)

        # --- Draw Left Column (Images) ---
        current_y_left = popup_y + POPUP_PADDING
        # Draw Google Logo
        if google_logo_resized is not None:
            logo_start_x = popup_x + POPUP_PADDING
            h, w = google_logo_resized.shape[:2]
            overlay[current_y_left:current_y_left + h, logo_start_x:logo_start_x + w] = google_logo_resized[:,:,:3] # Use only BGR channels
            current_y_left += h + POPUP_PADDING

        # Draw Map Pin
        if map_pin_resized is not None:
            pin_start_x = popup_x + POPUP_PADDING + (left_col_width - pin_w) // 2 # Center pin horizontally in left col
            h, w = map_pin_resized.shape[:2]
            overlay[current_y_left:current_y_left + h, pin_start_x:pin_start_x + w] = map_pin_resized[:,:,:3] # Use only BGR channels

        # --- Draw Right Column (Text) ---
        text_start_x = popup_x + POPUP_PADDING + left_col_width + POPUP_PADDING
        current_y_text = popup_y + POPUP_PADDING + line_height # Start baseline for first text line
        for line in text_lines:
            cv2.putText(overlay, line, (text_start_x, current_y_text), cv2.FONT_HERSHEY_SIMPLEX,
                        POPUP_TEXT_SCALE, POPUP_TEXT_COLOR, POPUP_TEXT_THICKNESS, lineType=cv2.LINE_AA)
            current_y_text += line_height + text_line_spacing # Move down for the next line

        # Blend overlay with original image
        img = cv2.addWeighted(overlay, POPUP_ALPHA, img, 1 - POPUP_ALPHA, 0)

    except Exception as e:
        print(f"Error drawing popup: {e}") # Catch potential errors during drawing/slicing

    # --- End Bottom Left Popup ---

    cv2.imshow("Garbage Collection Detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Resources released.")