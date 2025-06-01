import numpy as np
import cv2
import mediapipe as mp
import pyautogui
import time
import matplotlib.pyplot as plt

from PIL import Image

print("""
Virtual Mouse Instructions:

1.  Move your hand to move the mouse cursor.
2.  Click by moving your index finger up and down within a large distance.
3.  Drag by moving your thumb and index fingers apart.
4.  Take a screenshot by bringing your thumb and index fingers close together.

Press small Letter 'o' to start the program.
""")

# Wait for the users to press 'o'
while True:
    if input("Press 'o' to start the program: ") == 'o':
        break

# Initialize the MediaPipe hands detector
mp_hands = mp.solutions.hands
hand_detector = mp_hands.Hands(max_num_hands=1)
drawing_utils = mp.solutions.drawing_utils
HandLandmark = mp_hands.HandLandmark

# Get the size of the screen
screen_width, screen_height = pyautogui.size()

# Define constants
SCREENSHOT_DISTANCE_THRESHOLD = 30
CLICK_DISTANCE_THRESHOLD = 50  # Adjust as needed
DRAGGING_THRESHOLD = 50  # Adjust as needed
SCROLL_SENSITIVITY = 10  # Adjust as needed

# Initialize the VideoCapture object to use the webcam
cap = cv2.VideoCapture(0)

# Set the resolution of the webcam feed
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Initialize variables
dragging = False
previous_index_x, previous_index_y = 0, 0
scroll_direction = 0
thumb_x, thumb_y = 0, 0
index_x, index_y = 0, 0
last_index_y = 0

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab a frame from the camera.")
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with the hand detector
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks

    if hands:
        for hand in hands:
            # Draw hand landmarks (optimized using cv2.polylines)
            landmarks = hand.landmark
            points = [(int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])) for landmark in landmarks]
            points_array = np.array([points], dtype=np.int32)  # Convert points to a NumPy array
            cv2.polylines(frame, points_array, False, (0, 255, 0), 1)

            # Initialize thumb and index finger coordinates
            for id, landmark in enumerate(landmarks):
                # Calculate the coordinates
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])

                # Draw landmarks as red circles
                if id not in [4, 8]:
                    cv2.circle(frame, (x, y), 6, (0, 0, 255), -1)
                else:
                    cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)  # Yellow circle for screenshot and index finger landmarks

                # Check for the thumb tip
                if id == HandLandmark.THUMB_TIP:
                    thumb_x, thumb_y = screen_width / frame.shape[1] * x, screen_height / frame.shape[0] * y

                # Check for the index finger tip
                if id == HandLandmark.INDEX_FINGER_TIP:
                    index_x, index_y = screen_width / frame.shape[1] * x, screen_height / frame.shape[0] * y

            if 'previous_index_x' in locals() and 'previous_index_y' in locals():
                distance_y = abs(previous_index_y - index_y)

                # Check for click (if index finger Y movement is large)
                if distance_y > CLICK_DISTANCE_THRESHOLD and not dragging:
                    pyautogui.click()
                previous_index_x, previous_index_y = index_x, index_y  # Update previous coordinates for next frame
                if not dragging:
                    pyautogui.moveTo(index_x, index_y)

                # Check for dragging (if thumb and index fingers are apart)
                distance = np.sqrt((thumb_x - index_x) ** 2 + (thumb_y - index_y) ** 2)
                if distance > DRAGGING_THRESHOLD:
                    if not dragging:
                        pyautogui.mouseDown()
                        dragging = True
                else:
                    if dragging:
                        pyautogui.mouseUp()
                        dragging = False
                if dragging:
                    pyautogui.moveTo(index_x, index_y)

                # Check for screenshot (if thumb and index fingers are close) every 10th frame
                if frame.shape[0] % 10 == 0:
                    distance = np.sqrt((thumb_x - index_x) ** 2 + (thumb_y - index_y) ** 2)
                    if distance < SCREENSHOT_DISTANCE_THRESHOLD:
                        cv2.imwrite('screenshot.png', frame)
                        print("Screenshot taken!")

                        # Open the screenshot image using PIL
                        img = Image.open('screenshot.png')

                        # Display the image in the terminal using matplotlib
                        plt.imshow(img)
                        plt.show()

    # Display the frame
    cv2.imshow('Virtual Mouse', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
