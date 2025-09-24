import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

class SpectroscopyClassifier:
    def __init__(self, n_features=18):
        self.n_features = n_features
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.class_names = None
        
    def load_and_prepare_data(self, csv_file):
        """Load spectroscopy dataset from CSV"""
        print("Loading spectroscopy dataset...")
        
        try:
            # Load CSV data
            df = pd.read_csv(csv_file)
            print(f"Dataset shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            
            # Display first few rows
            print("\nFirst 5 rows:")
            print(df.head())
            
            # FIXED: Identify feature and target columns correctly
            # First column is target (plastic type), rest are features (wavelengths)
            target_column = df.columns[0]  # 'label'
            feature_columns = df.columns[1:].tolist()  # wavelength columns
            
            print(f"\nTarget column: {target_column}")
            print(f"Feature columns ({len(feature_columns)}): {feature_columns[:5]}...{feature_columns[-5:]}")
            
            # Extract features and target
            X = df[feature_columns].values.astype(np.float32)  # FIXED: Ensure float type
            y = df[target_column].values
            
            print(f"\nFeatures shape: {X.shape}")
            print(f"Target shape: {y.shape}")
            
            # Check for missing values - FIXED: Handle string target column
            print(f"Missing values in features: {np.isnan(X).sum()}")
            print(f"Missing values in target: {pd.isna(y).sum()}")
            
            # Display class distribution
            unique_classes = np.unique(y)
            print(f"\nUnique plastic types: {unique_classes}")
            for cls in unique_classes:
                count = np.sum(y == cls)
                print(f"  {cls}: {count} samples ({count/len(y)*100:.1f}%)")
            
            # Encode labels
            y_encoded = self.label_encoder.fit_transform(y)
            self.class_names = self.label_encoder.classes_
            print(f"\nEncoded classes: {dict(zip(range(len(self.class_names)), self.class_names))}")
            
            # Split data
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
            )
            
            print(f"\nData split:")
            print(f"  Training: {X_train.shape[0]} samples")
            print(f"  Validation: {X_val.shape[0]} samples")
            print(f"  Test: {X_test.shape[0]} samples")
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            X_test_scaled = self.scaler.transform(X_test)
            
            print(f"\nFeature scaling completed.")
            print(f"Training features range: [{X_train_scaled.min():.3f}, {X_train_scaled.max():.3f}]")
            
            return (X_train_scaled, y_train), (X_val_scaled, y_val), (X_test_scaled, y_test)
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("\nMake sure your CSV file has:")
            print("- First column: plastic types (PET, PS, PP, etc.)")
            print("- Remaining columns: wavelength readings (410nm, 430nm, ..., 940nm)")
            print("- No missing values")
            raise
    
    def build_model(self, n_classes):
        """Build lightweight MLP suitable for ESP32-S3"""
        print(f"\nBuilding MLP model for {n_classes} classes...")
        
        self.model = keras.Sequential([
            layers.Input(shape=(self.n_features,)),
            
            # First hidden layer
            layers.Dense(32, activation='relu', name='hidden1'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Second hidden layer
            layers.Dense(24, activation='relu', name='hidden2'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            # Third hidden layer (smaller for ESP32)
            layers.Dense(16, activation='relu', name='hidden3'),
            layers.Dropout(0.2),
            
            # Output layer
            layers.Dense(n_classes, activation='softmax', name='output')
        ])
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Model Architecture:")
        self.model.summary()
        
        # Calculate model parameters
        total_params = self.model.count_params()
        print(f"\nTotal parameters: {total_params:,}")
        print("Model is ready for training!")
        
        return self.model
    
    def train_model(self, train_data, val_data, epochs=50):
        """Train the MLP model"""
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        print(f"\nTraining spectroscopy model...")
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Validation samples: {X_val.shape[0]}")
        print(f"Features: {X_train.shape[1]}")
        
        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1,
            mode='max'
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        
        checkpoint = keras.callbacks.ModelCheckpoint(
            'best_spectroscopy_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr, checkpoint],
            verbose=1
        )
        
        return history
    
    def evaluate_model(self, test_data):
        """Evaluate model performance"""
        X_test, y_test = test_data
        
        print("\nEvaluating spectroscopy model...")
        
        # Get predictions
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=1)
        
        # Detailed predictions
        y_pred_probs = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        print(f"\nTest Results:")
        print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"Test Loss: {test_loss:.4f}")
        
        # Classification report
        print(f"\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred, target_names=self.class_names))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names, yticklabels=self.class_names)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix - Plastic Type Classification")
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
        
        # Show sample predictions
        print(f"\nSample Predictions:")
        for i in range(min(10, len(y_test))):
            true_class = self.class_names[y_test[i]]
            pred_class = self.class_names[y_pred[i]]
            confidence = np.max(y_pred_probs[i])
            status = "CORRECT" if y_test[i] == y_pred[i] else "WRONG"
            print(f"  {status} Sample {i+1}: True={true_class}, Predicted={pred_class}, Confidence={confidence:.3f}")
        
        return test_accuracy
    
    def convert_to_tflite(self, model_name="spectroscopy_mlp"):
        """Convert model to TensorFlow Lite"""
        print("\nConverting to TensorFlow Lite...")
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Convert
        tflite_model = converter.convert()
        
        # Save TFLite model
        tflite_filename = f"{model_name}.tflite"
        with open(tflite_filename, 'wb') as f:
            f.write(tflite_model)
        
        print(f"TFLite model saved as: {tflite_filename}")
        print(f"Model size: {len(tflite_model)/1024:.2f} KB")
        
        # Verify the model
        try:
            interpreter = tf.lite.Interpreter(model_content=tflite_model)
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            print(f"TFLite model verified successfully!")
            print(f"Input shape: {input_details[0]['shape']}")
            print(f"Output shape: {output_details[0]['shape']}")
            print(f"Input type: {input_details[0]['dtype']}")
            print(f"Output type: {output_details[0]['dtype']}")
            
            # Test with dummy input
            input_shape = input_details[0]['shape']
            dummy_input = np.random.random(input_shape).astype(np.float32)
            interpreter.set_tensor(input_details[0]['index'], dummy_input)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
            print(f"Test inference successful! Output shape: {output.shape}")
            
        except Exception as e:
            print(f"TFLite model verification failed: {e}")
        
        return tflite_filename
    
    def plot_training_history(self, history):
        """Plot training curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        epochs = range(1, len(history.history['accuracy']) + 1)
        
        # Plot accuracy
        ax1.plot(epochs, history.history['accuracy'], 'bo-', label='Training Accuracy', linewidth=2)
        ax1.plot(epochs, history.history['val_accuracy'], 'ro-', label='Validation Accuracy', linewidth=2)
        ax1.set_title('Model Accuracy - Spectroscopy Classification')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot loss
        ax2.plot(epochs, history.history['loss'], 'bo-', label='Training Loss', linewidth=2)
        ax2.plot(epochs, history.history['val_loss'], 'ro-', label='Validation Loss', linewidth=2)
        ax2.set_title('Model Loss - Spectroscopy Classification')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print training summary
        best_val_acc = max(history.history['val_accuracy'])
        best_val_acc_epoch = history.history['val_accuracy'].index(best_val_acc) + 1
        final_val_acc = history.history['val_accuracy'][-1]
        
        print(f"\nTraining Summary:")
        print(f"Best validation accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%) at epoch {best_val_acc_epoch}")
        print(f"Final validation accuracy: {final_val_acc:.4f} ({final_val_acc*100:.2f}%)")
        print(f"Total epochs trained: {len(epochs)}")
    
    def save_preprocessing_params(self, filename="spectroscopy_preprocessing.npz"):
        """Save scaler and label encoder for ESP32 deployment"""
        np.savez(filename,
                 scaler_mean=self.scaler.mean_,
                 scaler_scale=self.scaler.scale_,
                 class_names=self.class_names)
        print(f"Preprocessing parameters saved to: {filename}")
        print("Use these parameters in ESP32 for data normalization and label decoding")

# ---------------- Main Execution ----------------
def main():
    """Main training pipeline for spectroscopy classification"""
    print("Spectroscopy Plastic Classification - TinyML Training Pipeline")
    print("=" * 70)
    print("Target: ESP32-S3 Compatible Model for Competition Demo")
    print("=" * 70)
    
    # Initialize classifier
    classifier = SpectroscopyClassifier(n_features=18)
    
    # UPDATE THIS PATH to your spectroscopy CSV file
    csv_file = "E:/ml/microplastic/dataset/as7265x_synthetic_150.csv"
    
    try:
        # Load and prepare data
        print("Step 1: Loading spectroscopy dataset...")
        train_data, val_data, test_data = classifier.load_and_prepare_data(csv_file)
        
        # Build model
        print("\nStep 2: Building MLP model...")
        n_classes = len(classifier.class_names)
        classifier.build_model(n_classes)
        
        # Train model
        print("\nStep 3: Training model...")
        history = classifier.train_model(train_data, val_data, epochs=50)
        
        # Evaluate model
        print("\nStep 4: Evaluating model...")
        test_accuracy = classifier.evaluate_model(test_data)
        
        # Convert to TensorFlow Lite
        print("\nStep 5: Converting to TensorFlow Lite for ESP32-S3...")
        tflite_file = classifier.convert_to_tflite("spectroscopy_plastic_classifier")
        
        # Save preprocessing parameters
        print("\nStep 6: Saving preprocessing parameters...")
        classifier.save_preprocessing_params()
        
        # Plot training history
        print("\nStep 7: Analyzing training results...")
        classifier.plot_training_history(history)
        
        # Final summary
        print("\n" + "=" * 70)
        print("SPECTROSCOPY TRAINING PIPELINE COMPLETE!")
        print("=" * 70)
        print(f"Final Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"TFLite Model File: {tflite_file}")
        print(f"Number of Plastic Types: {n_classes}")
        print(f"Plastic Types: {', '.join(classifier.class_names)}")
        print(f"Model Status: Ready for ESP32-S3 deployment!")
        
        # Performance assessment
        if test_accuracy >= 0.90:
            print("EXCELLENT! Outstanding spectroscopy classification performance!")
        elif test_accuracy >= 0.85:
            print("GREAT! Very good performance for plastic type detection!")
        elif test_accuracy >= 0.75:
            print("GOOD! Should work well for demonstration.")
        else:
            print("FAIR! Consider more training data or hyperparameter tuning.")
        
        print(f"\nNext Steps:")
        print(f"1. Use {tflite_file} for ESP32-S3 deployment")
        print(f"2. Use spectroscopy_preprocessing.npz for data normalization")
        print(f"3. Test with your 18-band sensor readings!")
        print("=" * 70)
            
    except FileNotFoundError as e:
        print(f"Dataset Error: {e}")
        print("Please check your CSV file path and format.")
    except Exception as e:
        print(f"Training Error: {e}")
        print("Please check your CSV data format and try again.")

if __name__ == "__main__":
    main()