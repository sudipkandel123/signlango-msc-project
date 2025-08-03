import tensorflow as tf
import numpy as np


def create_test_model():
    """Create a simple test model for sign detection"""

    # Create a simple LSTM model
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(30, 1662)),  # 30 frames, 1662 features
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(
                4, activation="softmax"
            ),  # 4 classes: hi, please, excuse_me, okay
        ]
    )

    # Compile the model
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    # Save the model
    model.save("model.h5")
    model.save_weights("model.weights.h5")

    print("Test model created successfully!")
    print("Model summary:")
    model.summary()


if __name__ == "__main__":
    create_test_model()
