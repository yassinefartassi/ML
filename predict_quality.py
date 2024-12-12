import tensorflow as tf
import sys
import os
from urllib.parse import unquote

# Image parameters (same as in training)
IMG_HEIGHT = 224
IMG_WIDTH = 224

def predict_image_quality(model_path, image_path):
    """
    Predict the quality of a fruit/vegetable image
    
    Args:
        model_path: Path to the trained model (.h5 file)
        image_path: Path to the image to predict
    """
    try:
        # Handle spaces in filename
        image_path = unquote(image_path)
        
        # Check if image exists
        if not os.path.exists(image_path):
            print(f"Error: Image not found at {image_path}")
            return

        # Check if model exists
        if not os.path.exists(model_path):
            print(f"Error: Model not found at {model_path}")
            return

        # Load the model
        model = tf.keras.models.load_model(model_path)

        # Load and preprocess the image
        img = tf.keras.preprocessing.image.load_img(
            image_path,
            target_size=(IMG_HEIGHT, IMG_WIDTH)
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        img_array = img_array / 255.0

        # Make prediction
        prediction = model.predict(img_array, verbose=0)  # Added verbose=0 to reduce output
        quality = "Good" if prediction[0] > 0.5 else "Bad"
        confidence = float(prediction[0]) if prediction[0] > 0.5 else float(1 - prediction[0])
        
        print(f"\nPrediction Results:")
        print(f"Image: {os.path.basename(image_path)}")
        print(f"Quality: {quality}")
        print(f"Confidence: {confidence * 100:.2f}%")

    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_quality.py <path_to_image>")
        print('Example: python predict_quality.py "test image.jpg"')
        sys.exit(1)

    model_path = 'quality_detection_model.h5'
    # Join all arguments after the script name to handle filenames with spaces
    image_path = ' '.join(sys.argv[1:])
    predict_image_quality(model_path, image_path)
