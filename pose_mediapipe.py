import cv2
import mediapipe as mp
import math

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Function to process a single frame and display pose
def process_frame(frame):
    # Convert the frame to RGB (MediaPipe uses RGB format)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform pose estimation
    results = pose.process(frame_rgb)

    # Access the landmarks
    landmarks = results.pose_landmarks

    if landmarks:
        # Calculate the angle between the shoulders (assuming shoulder landmarks are at indices 11 and 12)
        shoulder_angle = angle_between_points(landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER],
                                              landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER])

        # Set the color based on the shoulder angle
        if abs(shoulder_angle) < 15:  # You can adjust this threshold as needed
            color = (0, 255, 0)  # Green for straight-looking faces
        else:
            color = (0, 0, 255)  # Red for non-straight-looking faces

        # Do something with the landmark data, e.g., draw lines or circles on the image
        # For simplicity, let's just draw lines connecting some of the keypoints
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing.draw_landmarks(frame, landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=color)

    # Return the annotated frame
    return frame

# Function to calculate the angle between two landmarks
def angle_between_points(point1, point2):
    dx = point2.x - point1.x
    dy = point2.y - point1.y
    radians = abs(math.atan2(dy, dx))
    return math.degrees(radians)

# Open the webcam
video_capture = cv2.VideoCapture(0)

while video_capture.isOpened():
    # Read a single frame
    ret, frame = video_capture.read()

    if not ret:
        break  # Break the loop if reading the frame fails

    # Process the frame
    processed_frame = process_frame(frame)

    # Display the frame with the pose estimation
    cv2.imshow('Real-time Pose Estimation', processed_frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
video_capture.release()
cv2.destroyAllWindows()
