import cv2 # type: ignore
import numpy as np # type: ignore
import tensorflow as tf # type: ignore
from tensorflow.keras.models import load_model # type: ignore

# Load the pre-trained model (make sure you have trained and saved the model as 'digit_recognition_cnn.h5')
model = load_model("digit_model.h5")

def preprocess_image(image):
    """ Preprocess the image for digit prediction. """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    resized = cv2.resize(gray, (28, 28))            # Resize to 28x28 pixels
    _, thresh = cv2.threshold(resized, 128, 255, cv2.THRESH_BINARY_INV)  # Binarize the image
    digit_normalized = thresh / 255.0               # Normalize to [0, 1]
    digit_ready = digit_normalized.reshape(1, 28, 28, 1)  # Reshape for the model
    return digit_ready

def predict_digit_from_image(image):
    """ Use the model to predict the digit from a preprocessed image. """
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_digit = np.argmax(prediction, axis=1)[0]
    confidence = prediction[0][predicted_digit]
    return predicted_digit, confidence

# Open a connection to the camera
cap = cv2.VideoCapture(0)  # 0 is the default camera, change if you have multiple cameras

if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Define a region of interest (ROI) for digit drawing/capture
    x, y, w, h = 100, 100, 200, 200
    roi = frame[y:y+h, x:x+w]

    # Predict the digit in the ROI
    try:
        digit, confidence = predict_digit_from_image(roi)
        cv2.putText(frame, f"Digit: {digit}, Conf: {confidence:.2f}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)
    except Exception as e:
        print(f"Prediction error: {e}")

    # Draw the ROI on the frame and display it
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imshow("Digit Recognition", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
