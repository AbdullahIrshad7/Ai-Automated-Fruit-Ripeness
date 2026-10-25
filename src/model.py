"""
Deep Learning CNN Model architecture and inference handler
for Automated Fruit Ripeness Detection & Classification.
"""

import os
import numpy as np
from PIL import Image
from src.utils import FRUIT_CLASSES, analyze_image_heuristics, get_fruit_knowledge

# Path to trained weights if downloaded locally
MODEL_FILENAME = "trained_model.h5"
POSSIBLE_PATHS = [
    MODEL_FILENAME,
    os.path.join(os.path.dirname(os.path.dirname(__file__)), MODEL_FILENAME),
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", MODEL_FILENAME),
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "Fruit Ripeness Final Submission", MODEL_FILENAME)
]

_CACHED_MODEL = None


def find_model_path():
    """Locate the trained model file if it exists."""
    for path in POSSIBLE_PATHS:
        if os.path.exists(path):
            return path
    return None


def is_tensorflow_available():
    """Check if TensorFlow is installed and importable."""
    try:
        import tensorflow as tf
        return True
    except ImportError:
        return False


def build_cnn_model(input_shape=(64, 64, 3), num_classes=131):
    """
    Constructs the exact sequential CNN architecture as trained in
    Training_Fruit_Recognition.ipynb:
    - Conv2D(64, 3x3, relu)
    - MaxPool2D(2x2, stride 2)
    - Conv2D(64, 3x3, relu)
    - MaxPool2D(2x2, stride 2)
    - Dropout(0.5)
    - Flatten
    - Dense(256, relu)
    - Dense(131, softmax)
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models

    model = models.Sequential([
        layers.Conv2D(64, kernel_size=3, activation='relu', input_shape=input_shape),
        layers.MaxPool2D(pool_size=2, strides=2),
        layers.Conv2D(64, kernel_size=3, activation='relu'),
        layers.MaxPool2D(pool_size=2, strides=2),
        layers.Dropout(0.5),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def load_trained_model():
    """Load cached or new trained Keras model if available."""
    global _CACHED_MODEL
    if _CACHED_MODEL is not None:
        return _CACHED_MODEL

    model_path = find_model_path()
    if model_path and is_tensorflow_available():
        import tensorflow as tf
        try:
            _CACHED_MODEL = tf.keras.models.load_model(model_path)
            return _CACHED_MODEL
        except Exception:
            return None
    return None


def predict_fruit(image: Image.Image) -> dict:
    """
    Predicts fruit class, variety, ripeness stage, and confidence.
    Uses Deep Learning CNN when weights are present, with seamless
    intelligent heuristic fallback for immediate browser/demo use.
    """
    model = load_trained_model()

    if model is not None and is_tensorflow_available():
        try:
            # Preprocess image as in training pipeline (64x64, normalized 0..1)
            img_resized = image.convert("RGB").resize((64, 64))
            img_array = np.array(img_resized, dtype=np.float32) / 255.0
            img_batch = np.expand_dims(img_array, axis=0)

            predictions = model.predict(img_batch, verbose=0)
            class_idx = int(np.argmax(predictions[0]))
            confidence = float(np.max(predictions[0])) * 100.0

            predicted_class = FRUIT_CLASSES[class_idx] if class_idx < len(FRUIT_CLASSES) else "Fruit Variety"

            # Parse ripeness state from class label
            class_lower = predicted_class.lower()
            if "not ripened" in class_lower or "green" in class_lower or "unripe" in class_lower:
                ripeness = "Unripe"
                badge = "🟡 Unripe (Maturing)"
            elif "ripe" in class_lower or confidence > 85.0:
                ripeness = "Ripe"
                badge = "🟢 Ripe (Optimal for Consumption)"
            else:
                ripeness = "Mature"
                badge = "🟢 Mature"

            fruit_base_name = predicted_class.split()[0]
            knowledge = get_fruit_knowledge(fruit_base_name)

            return {
                "fruit_name": fruit_base_name,
                "variety": predicted_class,
                "ripeness": ripeness,
                "ripeness_badge": badge,
                "confidence": round(confidence, 2),
                "freshness_score": round(min(confidence * 0.99, 99.0), 1),
                "storage": knowledge["storage"],
                "shelf_life": knowledge["shelf_life_ripe"] if ripeness == "Ripe" else knowledge["shelf_life_unripe"],
                "nutrients": knowledge["nutrients"],
                "ripeness_cues": knowledge["ripeness_cues"],
                "engine": "TensorFlow CNN (Deep Learning)"
            }
        except Exception:
            pass

    # Seamless Intelligent Heuristic Engine fallback
    result = analyze_image_heuristics(image)
    result["engine"] = "Computer Vision Heuristics (Instant Demo Engine)"
    return result
