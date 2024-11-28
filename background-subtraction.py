import cv2
import pickle
import numpy as np
from datetime import datetime

# Load video feed
cap = cv2.VideoCapture('video.mp4')
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Load background image of an empty parking lot
background_img = cv2.imread('empty_parking_lot.png')
if background_img is None:
    print("Error: Could not load background image.")
    exit()

# Ensure the background image matches the video frame size
background_img = cv2.resize(background_img, (frame_width, frame_height))

# Create VideoWriter object
# Generate filename with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f'parking_detection_{timestamp}.mp4'

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

# Load parking positions
try:
    with open('CarParkPosPoly', 'rb') as f:
        posList = pickle.load(f)
    if not posList:
        print("Warning: posList is empty. Ensure 'CarParkPosPoly' contains valid data.")
except FileNotFoundError:
    print("Error: File 'CarParkPosPoly' not found.")
    exit()
except Exception as e:
    print(f"Error loading 'CarParkPosPoly': {e}")
    exit()


def create_masks(frame_shape, posList):
    masks = []
    for poly in posList:
        mask = np.zeros(frame_shape[:2], dtype=np.uint8)
        pts = np.array(poly, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
        masks.append(mask)
    return masks


def checkParkingSpace(img, background_img, masks, posList):
    spaceCounter = 0
    img_diff = cv2.absdiff(background_img, img)
    img_gray_diff = cv2.cvtColor(img_diff, cv2.COLOR_BGR2GRAY)

    for idx, (mask, poly) in enumerate(zip(masks, posList)):
        imgMasked = cv2.bitwise_and(img_gray_diff, img_gray_diff, mask=mask)
        _, thresh = cv2.threshold(imgMasked, 50, 255, cv2.THRESH_BINARY)

        # Count non-zero pixels in the thresholded image
        non_zero_count = cv2.countNonZero(thresh)

        # Determine if parking space is empty or filled
        if non_zero_count < 2000:
            color = (0, 255, 0)
            spaceCounter += 1
        else:
            color = (0, 0, 255)

        # Draw polygon and text
        contours_poly = np.array(poly, dtype=np.int32)
        cv2.polylines(img, [contours_poly], isClosed=True, color=color, thickness=3, lineType=cv2.LINE_AA)

        M = cv2.moments(contours_poly)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            x, y, w, h = cv2.boundingRect(contours_poly)
            cX = x + w // 2
            cY = y + h // 2

        # Display the non-zero count
        cv2.putText(img, str(non_zero_count), (cX - 20, cY + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    # Display number of free spaces
    cv2.putText(img, f'Free: {spaceCounter}/{len(masks)}', (100, 50), cv2.FONT_HERSHEY_SIMPLEX,
                2, (0, 0, 0), 2, cv2.LINE_AA)


# Create masks once since parking positions are static
masks = create_masks(background_img.shape, posList)

# Set window to be resizable
cv2.namedWindow("Parking Lot", cv2.WINDOW_NORMAL)

while True:
    success, img = cap.read()
    if not success:
        print("Video playback completed.")
        break

    # Check for free/occupied parking spaces using background subtraction
    checkParkingSpace(img, background_img, masks, posList)

    # Write the frame to output video
    out.write(img)

    # Display the main output
    cv2.imshow("Parking Lot", img)

    # Press 'q' to exit the loop
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release everything
cap.release()
cv2.destroyAllWindows()
