import cv2
import pickle
import cvzone
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
        print("Warning: posList kosong. Pastikan 'CarParkPosPoly' memuat data yang benar.")
except FileNotFoundError:
    print("Error: File 'CarParkPosPoly' tidak ditemukan.")
    exit()
except Exception as e:
    print(f"Error saat memuat 'CarParkPosPoly': {e}")
    exit()


def create_masks(frame_shape, posList):
    masks = []
    for poly in posList:
        mask = np.zeros(frame_shape[:2], dtype=np.uint8)
        pts = np.array(poly, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
        masks.append(mask)
    return masks


def checkParkingSpace(img, masks, posList):
    # Image preprocessing for parking space detection
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 25, 16)
    imgMedian = cv2.medianBlur(imgThreshold, 5)
    kernel = np.ones((3, 3), np.uint8)
    imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)

    spaceCounter = 0

    for idx, (mask, poly) in enumerate(zip(masks, posList)):
        imgMasked = cv2.bitwise_and(imgDilate, imgDilate, mask=mask)
        count = cv2.countNonZero(imgMasked)

        if count < 850:
            color = (0, 255, 0)
            spaceCounter += 1
        else:
            color = (0, 0, 255)

        # Menggambar poligon pada frame asli
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.polylines(img, contours, isClosed=True, color=color, thickness=3, lineType=cv2.LINE_AA)

        # Menghitung pusat poligon untuk menampilkan count
        M = cv2.moments(contours[0])
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            # Jika m00 adalah nol, gunakan titik tengah dari bounding rectangle
            x, y, w, h = cv2.boundingRect(contours[0])
            cX = x + w // 2
            cY = y + h // 2

        # Menampilkan nilai count di dalam poligon
        cv2.putText(img, str(count), (cX - 20, cY + 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2, cv2.LINE_AA)

    # Menampilkan jumlah tempat parkir kosong
    cvzone.putTextRect(img, f'Free: {spaceCounter}/{len(masks)}', (100, 50), scale=3,
                       thickness=2, offset=20, colorR=(0, 200, 0))


# Create masks once since parking positions are static
masks = create_masks((frame_height, frame_width), posList)

# Set window to be resizable
cv2.namedWindow("Parking Lot", cv2.WINDOW_NORMAL)

while True:
    success, img = cap.read()
    if not success:
        print("Video playback completed.")
        break

    # Check for free/occupied parking spaces using masks and draw on frame asli
    checkParkingSpace(img, masks, posList)

    # Write the frame to output video
    out.write(img)

    # Display the main output
    cv2.imshow("Parking Lot", img)

    # Press 'q' to exit the loop
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video saved sebagai {output_filename}")
