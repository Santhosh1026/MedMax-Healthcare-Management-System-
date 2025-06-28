import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

class skincancer:
    def SkinCancerPrediction(self, image_file):
        # Get current app directory (same folder as this file)
        app_dir = os.path.dirname(os.path.abspath(__file__))

        # Load model from app's "model" folder
        model_path = os.path.join(app_dir, 'model', 'Skin-Cancer-detection.h5')
        model = load_model(model_path)

        # Prepare "uploads" folder inside the app
        upload_dir = os.path.join(app_dir, 'uploads')
        os.makedirs(upload_dir, exist_ok=True)

        # Save uploaded file to the uploads folder
        file_path = os.path.join(upload_dir, image_file.name)
        with open(file_path, 'wb+') as f:
            for chunk in image_file.chunks():
                f.write(chunk)

        # Preprocess the image
        test_image = image.load_img(file_path, target_size=(256, 256))
        test_image = image.img_to_array(test_image)
        test_image = test_image / 255.0
        test_image = np.expand_dims(test_image, axis=0)

        # Make prediction
        preds = model.predict(test_image)
        predicted_class = np.argmax(preds, axis=1)[0]

        # Map class index to label
        labels = {
            0: "The Predicted Skin Disease is Actinic Keratoses and Intraepithelial Carcinoma",
            1: "The Predicted Skin Disease is Basal Cell Carcinoma",
            2: "The Predicted Skin Disease is Benign Keratosis-like Lesions",
            3: "The Predicted Skin Disease is Dermatofibroma",
            4: "The Predicted Skin Disease is Melanoma",
            5: "The Predicted Skin Disease is Melanocytic Nevi",
            5: "The Predicted Skin Disease is Vascular Lesions",
        }

        return labels.get(predicted_class, "Unknown Prediction")
