#!/usr/bin/env python3
"""
Sign Language Model Training Script
==================================

This script trains an LSTM model for British Sign Language detection using
the existing MediaPipe keypoint data.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SignLanguageTrainer:
    def __init__(self, data_path="../MP_Data_HighAccuracy", model_save_path="./"):
        """
        Initialize the trainer with data path and model save path
        
        Args:
            data_path (str): Path to the MediaPipe data directory
            model_save_path (str): Path to save the trained model
        """
        self.data_path = data_path
        self.model_save_path = model_save_path
        self.signs = ['hi', 'please', 'excuse_me', 'okay']
        self.sequence_length = 30  # Number of frames per sequence
        self.feature_dim = 1662    # MediaPipe holistic features dimension
        
        # Training parameters
        self.batch_size = 32
        self.epochs = 100
        self.learning_rate = 0.001
        self.validation_split = 0.2
        
        # Data storage
        self.X = []
        self.y = []
        self.label_map = {sign: idx for idx, sign in enumerate(self.signs)}
        
        logger.info(f"Initialized trainer with signs: {self.signs}")
        logger.info(f"Data path: {self.data_path}")
        logger.info(f"Model save path: {self.model_save_path}")

    def load_data(self):
        """Load and preprocess the MediaPipe keypoint data"""
        logger.info("Loading training data...")
        
        for sign in self.signs:
            sign_path = os.path.join(self.data_path, sign)
            if not os.path.exists(sign_path):
                logger.warning(f"Sign directory not found: {sign_path}")
                continue
                
            logger.info(f"Loading data for sign: {sign}")
            sign_data = []
            
            # Get all sequence directories for this sign
            sequence_dirs = [d for d in os.listdir(sign_path) 
                           if os.path.isdir(os.path.join(sign_path, d))]
            
            for seq_dir in sequence_dirs:
                seq_path = os.path.join(sign_path, seq_dir)
                frame_files = [f for f in os.listdir(seq_path) if f.endswith('.npy')]
                frame_files.sort(key=lambda x: int(x.split('.')[0]))  # Sort by frame number
                
                if len(frame_files) >= self.sequence_length:
                    # Load sequence of frames
                    sequence = []
                    for i in range(self.sequence_length):
                        frame_file = frame_files[i]
                        frame_path = os.path.join(seq_path, frame_file)
                        frame_data = np.load(frame_path)
                        sequence.append(frame_data)
                    
                    # Pad or truncate to exact sequence length
                    if len(sequence) < self.sequence_length:
                        # Pad with last frame
                        last_frame = sequence[-1]
                        while len(sequence) < self.sequence_length:
                            sequence.append(last_frame)
                    elif len(sequence) > self.sequence_length:
                        # Truncate to sequence length
                        sequence = sequence[:self.sequence_length]
                    
                    sign_data.append(np.array(sequence))
            
            if sign_data:
                # Add to training data
                self.X.extend(sign_data)
                self.y.extend([self.label_map[sign]] * len(sign_data))
                logger.info(f"Loaded {len(sign_data)} sequences for sign '{sign}'")
            else:
                logger.warning(f"No data found for sign '{sign}'")
        
        # Convert to numpy arrays
        self.X = np.array(self.X)
        self.y = np.array(self.y)
        
        # One-hot encode labels
        self.y = tf.keras.utils.to_categorical(self.y, num_classes=len(self.signs))
        
        logger.info(f"Total sequences loaded: {len(self.X)}")
        logger.info(f"X shape: {self.X.shape}")
        logger.info(f"y shape: {self.y.shape}")
        
        return len(self.X) > 0

    def create_model(self):
        """Create the LSTM model architecture"""
        logger.info("Creating LSTM model...")
        
        model = Sequential([
            # First LSTM layer
            LSTM(128, return_sequences=True, 
                 input_shape=(self.sequence_length, self.feature_dim),
                 dropout=0.2, recurrent_dropout=0.2),
            
            # Second LSTM layer
            LSTM(64, return_sequences=True,
                 dropout=0.2, recurrent_dropout=0.2),
            
            # Third LSTM layer
            LSTM(32, dropout=0.2, recurrent_dropout=0.2),
            
            # Dense layers
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(len(self.signs), activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        logger.info("Model architecture:")
        model.summary()
        
        return model

    def train_model(self, model):
        """Train the model with callbacks and monitoring"""
        logger.info("Starting model training...")
        
        # Split data into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            self.X, self.y, test_size=self.validation_split, 
            random_state=42, stratify=self.y
        )
        
        logger.info(f"Training samples: {len(X_train)}")
        logger.info(f"Validation samples: {len(X_val)}")
        
        # Callbacks
        callbacks = [
            # Save best model
            ModelCheckpoint(
                filepath=os.path.join(self.model_save_path, 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            
            # Early stopping
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train the model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        return history, model

    def evaluate_model(self, model, X_test=None, y_test=None):
        """Evaluate the trained model"""
        logger.info("Evaluating model...")
        
        if X_test is None or y_test is None:
            # Use validation split for evaluation
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, test_size=self.validation_split, 
                random_state=42, stratify=self.y
            )
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(y_test, axis=1)
        
        # Calculate metrics
        test_loss, test_accuracy, test_precision, test_recall = model.evaluate(X_test, y_test, verbose=0)
        
        logger.info(f"Test Accuracy: {test_accuracy:.4f}")
        logger.info(f"Test Precision: {test_precision:.4f}")
        logger.info(f"Test Recall: {test_recall:.4f}")
        
        # Classification report
        report = classification_report(y_test_classes, y_pred_classes, 
                                     target_names=self.signs, output_dict=True)
        
        logger.info("Classification Report:")
        logger.info(classification_report(y_test_classes, y_pred_classes, 
                                        target_names=self.signs))
        
        # Confusion matrix
        cm = confusion_matrix(y_test_classes, y_pred_classes)
        
        return {
            'accuracy': test_accuracy,
            'precision': test_precision,
            'recall': test_recall,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'true_labels': y_test_classes,
            'predicted_labels': y_pred_classes
        }

    def plot_training_history(self, history):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(history.history['loss'], label='Training Loss')
        axes[0, 1].plot(history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision
        axes[1, 0].plot(history.history['precision'], label='Training Precision')
        axes[1, 0].plot(history.history['val_precision'], label='Validation Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Recall
        axes[1, 1].plot(history.history['recall'], label='Training Recall')
        axes[1, 1].plot(history.history['val_recall'], label='Validation Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_save_path, 'training_history.png'), dpi=300, bbox_inches='tight')
        plt.show()

    def plot_confusion_matrix(self, cm):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.signs, yticklabels=self.signs)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_save_path, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.show()

    def save_model_info(self, model, history, evaluation_results):
        """Save model information and training results"""
        model_info = {
            'training_date': datetime.now().isoformat(),
            'signs': self.signs,
            'sequence_length': self.sequence_length,
            'feature_dim': self.feature_dim,
            'model_architecture': {
                'layers': [layer.name for layer in model.layers],
                'total_params': model.count_params(),
                'trainable_params': sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
            },
            'training_parameters': {
                'batch_size': self.batch_size,
                'epochs': self.epochs,
                'learning_rate': self.learning_rate,
                'validation_split': self.validation_split
            },
            'training_results': {
                'final_accuracy': history.history['accuracy'][-1],
                'final_val_accuracy': history.history['val_accuracy'][-1],
                'best_val_accuracy': max(history.history['val_accuracy']),
                'epochs_trained': len(history.history['accuracy'])
            },
            'evaluation_results': {
                'test_accuracy': float(evaluation_results['accuracy']),
                'test_precision': float(evaluation_results['precision']),
                'test_recall': float(evaluation_results['recall']),
                'classification_report': evaluation_results['classification_report']
            }
        }
        
        # Save model info
        with open(os.path.join(self.model_save_path, 'model_info.json'), 'w') as f:
            json.dump(model_info, f, indent=2)
        
        logger.info("Model information saved to model_info.json")

    def run_training(self):
        """Run the complete training pipeline"""
        logger.info("Starting complete training pipeline...")
        
        # Step 1: Load data
        if not self.load_data():
            logger.error("Failed to load training data!")
            return False
        
        # Step 2: Create model
        model = self.create_model()
        
        # Step 3: Train model
        history, trained_model = self.train_model(model)
        
        # Step 4: Evaluate model
        evaluation_results = self.evaluate_model(trained_model)
        
        # Step 5: Plot results
        self.plot_training_history(history)
        self.plot_confusion_matrix(evaluation_results['confusion_matrix'])
        
        # Step 6: Save model and info
        trained_model.save(os.path.join(self.model_save_path, 'trained_model.h5'))
        self.save_model_info(trained_model, history, evaluation_results)
        
        logger.info("Training completed successfully!")
        logger.info(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
        logger.info(f"Test accuracy: {evaluation_results['accuracy']:.4f}")
        
        return True

def main():
    """Main function to run the training"""
    # Initialize trainer
    trainer = SignLanguageTrainer()
    
    # Run training
    success = trainer.run_training()
    
    if success:
        print("✅ Training completed successfully!")
        print("📁 Model files saved in the backend directory")
        print("📊 Training plots and evaluation results generated")
    else:
        print("❌ Training failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 