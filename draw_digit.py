# draw_digit.py
import tkinter as tk
import numpy as np # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from PIL import Image, ImageDraw, ImageTk # type: ignore

# Load the trained model
model = load_model('digit_model.h5')

# Function to preprocess the image from canvas for prediction
def preprocess_image(img):
    img = img.resize((28, 28)).convert('L')  # Resize to 28x28 and grayscale
    img = np.array(img)
    img = 255 - img  # Invert colors (black on white background)
    img = img / 255.0  # Normalize
    img = img.reshape(1, 28, 28, 1)  # Reshape for model input
    return img

# Function to clear the canvas
def clear_canvas():
    canvas.delete("all")
    global img, draw
    img = Image.new("L", (280, 280), color="white")
    draw = ImageDraw.Draw(img)

# Function to predict the digit
def predict_digit():
    processed_img = preprocess_image(img)
    prediction = model.predict(processed_img)
    predicted_digit = np.argmax(prediction)
    result_label.config(text=f"Predicted Digit: {predicted_digit}")

# Initialize GUI window
window = tk.Tk()
window.title("Draw a Digit")
window.geometry("350x450")
window.configure(bg="#f7f7f7")

# Create canvas for drawing
canvas = tk.Canvas(window, width=280, height=280, bg="white", bd=2, relief="solid")
canvas.pack(pady=10)

# Create initial blank image
img = Image.new("L", (280, 280), color="white")
draw = ImageDraw.Draw(img)

# Draw on the canvas
def draw_on_canvas(event):
    x1, y1 = (event.x - 5), (event.y - 5)
    x2, y2 = (event.x + 5), (event.y + 5)
    canvas.create_oval(x1, y1, x2, y2, fill="black", width=10)
    draw.line([x1, y1, x2, y2], fill="black", width=10)

canvas.bind("<B1-Motion>", draw_on_canvas)

# Add buttons for clearing and predicting
clear_button = tk.Button(window, text="Clear", command=clear_canvas, bg="#4CAF50", fg="white", font=("Arial", 14), relief="flat", width=10)
clear_button.pack(pady=10)

predict_button = tk.Button(window, text="Predict", command=predict_digit, bg="#4CAF50", fg="white", font=("Arial", 14), relief="flat", width=10)
predict_button.pack(pady=10)

# Label to display prediction result
result_label = tk.Label(window, text="Predicted Digit: ", font=("Arial", 16, "bold"), bg="#f7f7f7")
result_label.pack(pady=10)

# Run the Tkinter event loop
window.mainloop()
