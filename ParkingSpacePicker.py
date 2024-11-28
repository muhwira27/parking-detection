import cv2
import pickle
import numpy as np

# Constants
WINDOW_NAME = "Image"
DATA_FILE = 'CarParkPosPoly'
POINTS_PER_SPACE = 4

# Initialize variables
drawing = False
current_points = []
posList = []

# Load existing parking positions if available
try:
    with open(DATA_FILE, 'rb') as f:
        posList = pickle.load(f)
except Exception as e:
    posList = []


def mouseClick(event, x, y, flags, params):
    global drawing, current_points, posList

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(current_points) < POINTS_PER_SPACE:
            current_points.append((x, y))
            print(f'Point {len(current_points)} recorded: ({x}, {y})')
            if len(current_points) == POINTS_PER_SPACE:
                posList.append(current_points.copy())
                current_points = []
                print('Parking space added.')
                savePosList()

    elif event == cv2.EVENT_RBUTTONDOWN:
        for i, poly in enumerate(posList):
            # Create a contour from the polygon points
            contour = np.array(poly, dtype=np.int32)
            if cv2.pointPolygonTest(contour, (x, y), False) >= 0:
                posList.pop(i)
                print('Parking space removed.')
                savePosList()
                break


def savePosList():
    with open(DATA_FILE, 'wb') as f:
        pickle.dump(posList, f)
        print(f'Saved {len(posList)} parking spaces.')


def drawAllSpaces(img):
    overlay = img.copy()
    for poly in posList:
        # Draw the polygon lines
        cv2.polylines(img, [np.array(poly, dtype=np.int32)], isClosed=True, color=(255, 0, 255), thickness=2)
        # Fill the polygon with transparency
        cv2.fillPoly(overlay, [np.array(poly, dtype=np.int32)], color=(255, 0, 255))
    alpha = 0.2  # Transparency factor
    # Blend the overlay with the original image
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    return img


# Allow the window to be resizable
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(WINDOW_NAME, mouseClick)

while True:
    img = cv2.imread('img.png')
    if img is None:
        print('Error: Gambar tidak ditemukan atau gagal dimuat.')
        break
    img = drawAllSpaces(img)

    # Draw current polygon points
    if len(current_points) > 0:
        for point in current_points:
            cv2.circle(img, point, 5, (0, 255, 0), cv2.FILLED)
        # Draw lines between points
        for i in range(1, len(current_points)):
            cv2.line(img, current_points[i - 1], current_points[i], (0, 255, 0), 2)

    cv2.imshow(WINDOW_NAME, img)

    # Instructions for the user
    print("Left-click to select points for a parking space. Right-click inside a space to remove it.")
    print("Press 'c' to clear current points, or 'q' to quit.")

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        current_points = []
        print('Current points cleared.')
    elif key == ord('q'):
        break

cv2.destroyAllWindows()
