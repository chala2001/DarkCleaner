import tensorflow as tf
import os
import sys
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('skin_type_model.h5')

# Define the class labels (the order should match the folder structure in your training data)
class_labels = ['dry', 'normal', 'oily']

# Function to preprocess and predict on a new image
def predict_skin_type(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(128, 128))  # Match the size used during training
    img_array = image.img_to_array(img)  # Convert the image to a numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Rescale the pixel values to [0, 1]

    # Make the prediction
    predictions = model.predict(img_array)

    # Get the predicted class (index with highest probability)
    predicted_class_index = np.argmax(predictions)

    # Map the predicted class index to the corresponding label
    predicted_class = class_labels[predicted_class_index]

    return predicted_class

# Function to predict skin type for images given a list of image paths
def predict_on_images(image_paths):
    for img_path in image_paths:
        # Get the predicted skin type for the image
        predicted_class = predict_skin_type(img_path)
        
        # Print the image name and predicted class
        print(f"Image: {img_path} | Predicted Label: {predicted_class}")

# If input is passed from Java or command line (via system arguments)
if __name__ == "__main__":
    if len(sys.argv) > 1:
        # If image paths are provided as arguments
        image_paths = sys.argv[1:]  # List of image paths passed from Java (or command line)
    else:
        # Or, if Java sends the paths dynamically, you can read from stdin or a predefined list
        print("No image paths provided.")
        sys.exit(1)

    # Call the function to predict on the provided image paths
    predict_on_images(image_paths)
