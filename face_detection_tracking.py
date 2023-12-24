import cv2

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create a multi-object tracker (using KCF tracker)
tracker = cv2.MultiTracker_create()

# Open the default camera (you may need to change the argument if using an external camera)
cap = cv2.VideoCapture(0)

# Initialize face ID counter
face_id_counter = 0
faces_info = {}  # Dictionary to store face information

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # Update the tracker with the new frame and bounding boxes
    for face_id, bbox in tracker.update(frame):
        x, y, w, h = map(int, bbox)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"Face ID: {face_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Update the tracker with the new faces
    for (x, y, w, h) in faces:
        bbox = (x, y, w, h)
        # Check if the face is already being tracked
        face_tracked = False
        for face_id, face_info in faces_info.items():
            if cv2.RectIntersect(face_info["bbox"], bbox):
                tracker.add(cv2.TrackerKCF_create(), frame, bbox)
                face_tracked = True
                break

        # If the face is not being tracked, assign a new face ID and start tracking
        if not face_tracked:
            face_id_counter += 1
            tracker.add(cv2.TrackerKCF_create(), frame, bbox)
            faces_info[face_id_counter] = {"bbox": bbox}

    # Display the resulting frame
    cv2.imshow('Face Tracking', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
