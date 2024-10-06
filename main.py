import cv2
from fer import FER
import tkinter as tk
from tkinter import scrolledtext, Listbox, SINGLE, Toplevel
import threading
import requests
import json
from pydub import AudioSegment
from pydub.playback import play
import io
import time

# Initialize the FER model
# Using MTCNN for face detection
# FER is a facial emotion recognition library
# MTCNN is a face detector used by FER to detect faces in images
detector = FER(mtcnn=True)

# Boolean field to enable/disable OpenCV UI
enable_opencv_ui = False

# Initialize variables to capture emotion data
emotions_data = []
camera_index = -1

# Eleven Labs API details (Replace with your actual API key and voice ID)
ELEVEN_LABS_API_KEY = "sk_f3995b26128773eb8c55b03fc3a151074b16062a51af7502"
ELEVEN_LABS_VOICE_ID = "ZF6FPAbjXT4488VcRRnw"
ELEVEN_LABS_TTS_URL = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_LABS_VOICE_ID}"

# Emotion weights for adjusting dominance of each emotion
emotion_weights = {
    'neutral': 1.0,
    'happy': 1.0,
    'sad': 1.0,
    'angry': 1.0,
    'surprise': 1.0
}

# Capture flag for settings window
capturing_in_settings = False

# Function to capture emotions using the camera
def capture_emotions():
    global emotions_data
    cap = cv2.VideoCapture(camera_index)

    # Check if the camera is opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Continuously capture frames while capturing flag is True
    while capturing:
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Resize frame to speed up processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        # Convert frame to RGB color space for FER detection
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detect emotions in the frame
        result = detector.detect_emotions(rgb_frame)
        if result:
            emotions = result[0]["emotions"]
            emotions_data.append(emotions)

            # If OpenCV UI is enabled, display the detection results
            if enable_opencv_ui:
                bounding_box = result[0]["box"]
                (x, y, w, h) = [v * 2 for v in bounding_box]  # Scale bounding box back to original frame size
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                for i, (emotion, score) in enumerate(emotions.items()):
                    text = f"{emotion}: {score:.2f}"
                    cv2.putText(frame, text, (x, y - 10 - i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                                cv2.LINE_AA)

        # Display the frame with the bounding box and emotion scores
        if enable_opencv_ui:
            cv2.imshow('Emotion Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release the camera and close OpenCV windows
    cap.release()
    if enable_opencv_ui:
        cv2.destroyAllWindows()

# Function to capture emotions for settings adjustment
def capture_emotions_for_settings(selected_emotion):
    global capturing_in_settings
    cap = cv2.VideoCapture(camera_index)

    # Check if the camera is opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    emotions_list = []
    start_time = time.time()

    # Capture frames for 10 seconds
    while time.time() - start_time < 10:
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Resize frame to speed up processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        # Convert frame to RGB color space for FER detection
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detect emotions in the frame
        result = detector.detect_emotions(rgb_frame)
        if result:
            emotions = result[0]["emotions"]
            emotions_list.append(emotions)

    cap.release()

    # Adjust weights based on the captured emotions
    adjust_weights(selected_emotion, emotions_list)

# Function to adjust emotion weights based on captured data
def adjust_weights(selected_emotion, emotions_list):
    global emotion_weights

    # Calculate weighted average of detected emotions
    weighted_emotions = {emotion: 0 for emotion in emotions_list[0]}
    for emotions in emotions_list:
        for emotion, score in emotions.items():
            weighted_emotions[emotion] += score * emotion_weights[emotion]
    for emotion in weighted_emotions:
        weighted_emotions[emotion] /= len(emotions_list)

    # Determine the dominant emotion
    dominant_emotion = max(weighted_emotions, key=weighted_emotions.get)

    # Adjust weights to prioritize the selected emotion if it is not the dominant one
    if dominant_emotion != selected_emotion:
        emotion_weights[selected_emotion] *= 1.1  # Increase the weight of the selected emotion
        emotion_weights[dominant_emotion] *= 0.9  # Decrease the weight of the dominant emotion

    print(f"Adjusted weights: {emotion_weights}")

# Function to get the dominant emotion from a list of emotions
def get_dominant_emotion(emotions_list):
    # Calculate weighted average of detected emotions
    weighted_emotions = {emotion: 0 for emotion in emotions_list[0]}
    for emotions in emotions_list:
        for emotion, score in emotions.items():
            weighted_emotions[emotion] += score * emotion_weights[emotion]
    for emotion in weighted_emotions:
        weighted_emotions[emotion] /= len(emotions_list)
    return max(weighted_emotions, key=weighted_emotions.get)

# Function to generate TTS audio using Eleven Labs API
def generate_tts(text):
    headers = {
        "xi-api-key": ELEVEN_LABS_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "text": text,
        "model_id": "eleven_monolingual_v1",  # Adjust this as needed
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5,
            "style": 0.5,
            "use_speaker_boost": False
        }
    }
    # Make a POST request to the TTS API
    response = requests.post(ELEVEN_LABS_TTS_URL, headers=headers, json=payload)

    # Play the audio if the response is successful
    if response.status_code == 200:
        audio_data = response.content
        audio = AudioSegment.from_file(io.BytesIO(audio_data), format="mp3")
        play(audio)
    else:
        print(f"Error: {response.status_code}, {response.text}")

# Callback function for the generate button
def on_generate():
    global capturing, emotions_data
    capturing = False
    text = text_area.get("1.0", tk.END).strip()
    if text:
        log_output(text)
        print(f"Text: {text}")
        generate_tts(text)
    if emotions_data:
        dominant_emotion = get_dominant_emotion(emotions_data)
        print(f"Dominant emotion: {dominant_emotion}")
    else:
        print("No emotion data captured.")
    emotions_data = []

# Function to start capturing emotions when typing starts
def start_capture(event=None):
    global capturing
    capturing = True
    capture_thread = threading.Thread(target=capture_emotions)
    capture_thread.start()

# Function to find available cameras on the system
def find_available_cameras():
    available_cameras = []
    # Check the first 10 camera indexes to see which are available
    for index in range(10):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            available_cameras.append(index)
            cap.release()
    return available_cameras

# Function to select the first available camera
def select_camera():
    global camera_index

    available_cameras = find_available_cameras()
    if not available_cameras:
        print("No cameras found.")
        return False
    else:
        camera_index = available_cameras[0]  # Automatically select the first available camera
        return True

# Function to log output to the listbox in the UI
def log_output(output):
    log_listbox.insert(tk.END, output)

# Function to create the main UI
def create_ui():
    global log_listbox, text_area

    # Create the main window
    window = tk.Tk()
    window.title("Emotion Detection TTS")

    # Create a frame to hold the widgets
    frame = tk.Frame(window)
    frame.grid(column=0, row=0, padx=10, pady=10)

    # Text area for user input
    text_area = scrolledtext.ScrolledText(frame, wrap=tk.WORD, width=50, height=10)
    text_area.grid(column=0, row=0, padx=10, pady=10)

    # Listbox to log outputs
    log_listbox = Listbox(frame, selectmode=SINGLE, width=30, height=10)
    log_listbox.grid(column=1, row=0, padx=10, pady=10, rowspan=2, sticky="ns")

    # Button to generate TTS
    generate_button = tk.Button(frame, text="Generate", command=on_generate)
    generate_button.grid(column=0, row=1, padx=10, pady=10, sticky="w")

    # Button to open settings window
    settings_button = tk.Button(frame, text="Settings", command=open_settings_window)
    settings_button.grid(column=1, row=1, padx=10, pady=10, sticky="e")

    # Bind events to start emotion capture
    text_area.bind("<Return>", lambda event: on_generate())
    text_area.bind("<KeyPress>", start_capture)

    # Set fixed window size and center it
    window.update_idletasks()
    width = 800
    height = 400
    x = (window.winfo_screenwidth() // 2) - (width // 2)
    y = (window.winfo_screenheight() // 2) - (height // 2)
    window.geometry(f'{width}x{height}+{x}+{y}')
    window.resizable(False, False)

    # Run the main event loop
    window.mainloop()

# Function to open the settings window
def open_settings_window():
    global capturing_in_settings

    # Create a new window for settings
    settings_window = Toplevel()
    settings_window.title("Adjust Emotion Weights")

    # Create a frame to hold the settings widgets
    settings_frame = tk.Frame(settings_window)
    settings_frame.grid(column=0, row=0, padx=10, pady=10)

    # Label for emotion selection
    tk.Label(settings_frame, text="Select Emotion to Adjust").grid(column=0, row=0, columnspan=2)

    # Radio buttons for selecting an emotion to adjust
    emotions = ['neutral', 'happy', 'sad', 'angry', 'surprise']
    selected_emotion = tk.StringVar(value=emotions[0])

    for emotion in emotions:
        tk.Radiobutton(settings_frame, text=emotion.capitalize(), variable=selected_emotion, value=emotion).grid(
            column=0, row=emotions.index(emotion) + 1, sticky="w")

    # Button to adjust the selected emotion
    adjust_button = tk.Button(settings_frame, text="Adjust", command=lambda: adjust_emotion(selected_emotion.get()))
    adjust_button.grid(column=0, row=len(emotions) + 1, pady=10)

    # Function to handle closing the settings window
    def on_close():
        global capturing_in_settings
        capturing_in_settings = False
        settings_window.destroy()

    settings_window.protocol("WM_DELETE_WINDOW", on_close)

# Function to adjust the selected emotion
def adjust_emotion(selected_emotion):
    global capturing_in_settings
    capturing_in_settings = True
    # Start a new thread to capture emotions for settings adjustment
    capture_thread = threading.Thread(target=capture_emotions_for_settings, args=(selected_emotion,))
    capture_thread.start()

# Main function to run the application
if __name__ == "__main__":
    # Select the first available camera and create the UI if a camera is found
    if select_camera():
        create_ui()
    else:
        print("No camera selected. Exiting.")
