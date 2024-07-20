import cv2
from fer import FER

# Function to find the first available camera index
def find_available_camera_index(max_index=10):
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cap.release()
            return i
    return -1

# Find the available camera index
camera_index = find_available_camera_index()

if camera_index == -1:
    print("Error: No available camera found.")
    exit()

# Initialize the video capture
cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Initialize the FER model
detector = FER(mtcnn=True)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Resize the frame to speed up processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    # Convert the image to RGB (FER expects RGB images)
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect emotions
    result = detector.detect_emotions(rgb_frame)

    if result:
        # Extract emotions and bounding box
        bounding_box = result[0]["box"]
        emotions = result[0]["emotions"]

        # Scale the bounding box back to the original frame size
        (x, y, w, h) = [v * 2 for v in bounding_box]

        # Draw bounding box on the original frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Display the detected emotions on the frame
        for i, (emotion, score) in enumerate(emotions.items()):
            text = f"{emotion}: {score:.2f}"
            cv2.putText(frame, text, (x, y - 10 - i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Emotion Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the windows
cap.release()
cv2.destroyAllWindows()
