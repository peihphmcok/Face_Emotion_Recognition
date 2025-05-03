import numpy as np
import cv2
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Define emotions
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Load model
print("Loading model...")
classifier = load_model('model_78.h5')
classifier.load_weights("model_weights_78.weights.h5")
print("Model loaded!")

# Load face detector
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    print("Face detector loaded successfully")
except Exception as e:
    print(f"Error loading face detector: {e}")
    exit(1)


def process_frame(frame):
    # Convert to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.3,
        minNeighbors=5
    )

    # Process each face
    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

        # Extract face region
        roi_gray = gray_frame[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        # Prepare for prediction
        if np.sum(roi_gray) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # Make prediction
            prediction = classifier.predict(roi)[0]
            maxindex = int(np.argmax(prediction))
            emotion = emotion_labels[maxindex]

            # Display emotion on frame
            label_position = (x, y - 10)
            cv2.putText(frame, emotion, label_position,
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame


def main():
    print("Starting camera...")
    cap = cv2.VideoCapture(0)  # 0 is usually the default camera

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Camera is running. Press 'q' to quit.")

    while True:
        # Capture frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture image")
            break

        # Process and display
        output_frame = process_frame(frame)
        cv2.imshow('Face Emotion Recognition', output_frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Camera stopped")


if __name__ == "__main__":
    main()