import torch
import cv2
from mtcnn import MTCNN
from deep_sort_realtime.deepsort_tracker import DeepSort
import face_recognition
import os

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=10)

# Load pre-trained MTCNN model for face detection
mtcnn = MTCNN()

# Open video capture (change this to your camera index or video file)
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

# Size threshold for faces (adjust as needed)
min_face_size = 5000  # For example, minimum area of 5000 pixels

# Dictionary to track whether the face image and embedding have been saved for each ID
saved_data = {}

# Create a directory to save face images
os.makedirs("footfall(V1)", exist_ok=True)

# Initialize frame_id
frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Increment frame_id
    frame_id += 1

    # Detect faces using MTCNN
    faces = mtcnn.detect_faces(frame)

    # Draw bounding boxes for all detected faces
    for face in faces:
        box = face['box']
        confidence = face['confidence']

        # Draw bounding box
        cv2.rectangle(frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0, 255, 0), 2)

    # Update tracker with the detected faces
    bbs = [(face['box'], face['confidence'], face['keypoints']) for face in faces]
    tracks = tracker.update_tracks(bbs, frame=frame)

    # Draw bounding boxes and IDs on the frame
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id

        # Check if the face with the current track_id has already been saved
        if track_id in saved_data:
            continue

        ltrb = track.to_ltrb()

        # Check face size
        face_area = (ltrb[2] - ltrb[0]) * (ltrb[3] - ltrb[1])
        if face_area < min_face_size:
            continue  # Skip small faces

        # Get facial landmarks using face_recognition library
        face_locations = [(int(ltrb[1]), int(ltrb[2]), int(ltrb[3]), int(ltrb[0]))]  # (top, right, bottom, left)
        landmarks = face_recognition.face_landmarks(frame, face_locations)

        if landmarks:
            # Get face embedding
            face_encoding = face_recognition.face_encodings(frame, face_locations)[0]

            # Check if the face embedding is not empty before saving
            if len(face_encoding) > 0:
                # Compare with existing embeddings
                match = False
                for saved_id, saved_info in saved_data.items():
                    saved_embedding = saved_info["embedding"]
                    # Compare the face embeddings using face_recognition library
                    results = face_recognition.compare_faces([saved_embedding], face_encoding)

                    # Check if there is a match
                    if any(results):
                        match = True
                        break

                # If no match, save the face embedding
                if not match:
                    # Save the face embedding
                    saved_data[track_id] = {
                        "embedding": face_encoding,
                        "image_path": f"footfall(V1)/{track_id}_frame{frame_id}.png"
                    }

                    # Get the face bounding box coordinates
                    top, right, bottom, left = int(ltrb[1]), int(ltrb[2]), int(ltrb[3]), int(ltrb[0])

                    face_image = frame[top:bottom, left:right]

                    # Save the face image
                    cv2.imwrite(saved_data[track_id]["image_path"], face_image)

                    # Draw label
                    label = f"ID: {track_id}"
                    cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw count on the frame
    cv2.putText(frame, f"Unique Faces: {len(saved_data)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Real-Time Face Recognition", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()