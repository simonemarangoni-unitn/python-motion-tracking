# Import necessary libraries
import cv2
import numpy as np
import csv

# Path to video
video_path = "videos/helicopter3.mp4"
video = cv2.VideoCapture(video_path)

# Read only the first frame for drawing a rectangle for the desired object
ret, frame = video.read()

# Initialize coordinates for the rectangle
x_min, y_min, x_max, y_max = 36000, 36000, 0, 0

def coordinat_chooser(event, x, y, flags, param):
    global x_min, y_min, x_max, y_max

    # Handle left mouse button click
    if event == cv2.EVENT_LBUTTONDOWN:
        x_min = min(x, x_min)
        y_min = min(y, y_min)
        x_max = max(x, x_max)
        y_max = max(y, y_max)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)

    # Reset coordinates with the middle mouse button
    if event == cv2.EVENT_MBUTTONDOWN:
        print("Reset coordinate data")
        x_min, y_min, x_max, y_max = 36000, 36000, 0, 0

cv2.namedWindow('coordinate_screen')
cv2.setMouseCallback('coordinate_screen', coordinat_chooser)

while True:
    cv2.imshow("coordinate_screen", frame)  # Show only first frame
    k = cv2.waitKey(5) & 0xFF  # Press ESC to exit
    if k == 27:
        cv2.destroyAllWindows()
        break

# Take region of interest (inside of rectangle)
roi_image = frame[y_min:y_max, x_min:x_max]

# Convert ROI to grayscale, SIFT works with grayscale images
roi_gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)

# Step 2: Find key points of ROI (target image)
sift = cv2.SIFT_create()
keypoints_1, descriptors_1 = sift.detectAndCompute(roi_gray, None)

# Draw keypoints on the ROI image (optional visualization)
roi_keypoint_image = cv2.drawKeypoints(roi_gray, keypoints_1, roi_gray)

# Step 3: Track the target object in Video
video = cv2.VideoCapture(video_path)  # Reopen the video
bf = cv2.BFMatcher()

# Initialize statistics
total_matches = 0
frames_processed = 0


while True:
    ret, frame = video.read()
    if not ret:
        break  # Exit if there are no more frames

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    keypoints_2, descriptors_2 = sift.detectAndCompute(frame_gray, None)

    # Compare key points/descriptors
    matches = bf.match(descriptors_1, descriptors_2)
    total_matches += len(matches)
    frames_processed += 1

    for match in matches:
        query_idx = match.queryIdx
        train_idx = match.trainIdx
        pt2 = keypoints_2[train_idx].pt
        cv2.circle(frame, (int(pt2[0]), int(pt2[1])), 2, (255, 0, 0), 2)

    cv2.imshow("coordinate_screen", frame)

    k = cv2.waitKey(5) & 0xFF  # Press ESC to exit
    if k == 27:
        cv2.destroyAllWindows()
        break

# Write statistics to a CSV file
with open('statistiche_progetto.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Total Matches","Frames Processed"])
    writer.writerow([total_matches, frames_processed])

print("Statistiche salvate in 'statistiche_progetto.csv'.")
cv2.destroyAllWindows()
