import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

class MicroplasticImageClassifier:
    def __init__(self, img_height=64, img_width=64, batch_size=32):  # Increased batch size
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.model = None
        
    def load_dataset(self, data_dir):
        """Load image dataset from directory structure"""
        # Check if directories exist - handle both naming conventions
        train_dir = os.path.join(data_dir, '1_Training')
        val_dir = os.path.join(data_dir, '2_Validation') 
        test_dir = os.path.join(data_dir, '3_Testing')
        
        print("Checking directories...")
        for dir_path, dir_name in [(train_dir, '1_Training'), (val_dir, '2_Validation'), (test_dir, '3_Testing')]:
            if not os.path.exists(dir_path):
                raise FileNotFoundError(f"Directory not found: {dir_path}")
            print(f"Found {dir_name} directory")
        
        try:
            # Load datasets with shuffle=True for better training
            train_ds = tf.keras.utils.image_dataset_from_directory(
                train_dir,
                image_size=(self.img_height, self.img_width),
                batch_size=self.batch_size,
                class_names=['1_Clean_Water', '2_Microplastics'],
                shuffle=True,
                seed=123
            )
            
            val_ds = tf.keras.utils.image_dataset_from_directory(
                val_dir,
                image_size=(self.img_height, self.img_width),
                batch_size=self.batch_size,
                class_names=['1_Clean_Water', '2_Microplastics'],
                shuffle=False,
                seed=123
            )
            
            test_ds = tf.keras.utils.image_dataset_from_directory(
                test_dir,
                image_size=(self.img_height, self.img_width),
                batch_size=self.batch_size,
                class_names=['1_Clean_Water', '2_Microplastics'],
                shuffle=False,
                seed=123
            )
            
            # Print dataset info
            print("Datasets loaded successfully!")
            
            # More accurate count calculation
            def count_images_in_dataset(dataset):
                total = 0
                for batch in dataset:
                    total += batch[0].shape[0]
                return total
            
            train_count = count_images_in_dataset(train_ds)
            val_count = count_images_in_dataset(val_ds)
            test_count = count_images_in_dataset(test_ds)
            
            print("Dataset sizes:")
            print(f"   Training images: {train_count}")
            print(f"   Validation images: {val_count}")
            print(f"   Test images: {test_count}")
            print(f"   Total images: {train_count + val_count + test_count}")
            
            return train_ds, val_ds, test_ds
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("\nMake sure your directory structure is:")
            print("dataset/")
            print("├── 1_Training/")
            print("│   ├── 1_Clean_Water/")
            print("│   └── 2_Microplastics/")
            print("├── 2_Validation/")
            print("│   ├── 1_Clean_Water/") 
            print("│   └── 2_Microplastics/")
            print("└── 3_Testing/")
            print("    ├── 1_Clean_Water/")
            print("    └── 2_Microplastics/")
            raise
    
    def preprocess_dataset(self, dataset, augment=False):
        """Normalize pixel values and apply heavy augmentation to reduce overfitting"""
        
        # Very aggressive data augmentation to prevent overfitting
        if augment:
            data_augmentation = keras.Sequential([
                layers.RandomFlip("horizontal"),
                layers.RandomFlip("vertical"),
                layers.RandomRotation(0.3),        # Increased rotation
                layers.RandomZoom(0.25),           # Increased zoom
                layers.RandomContrast(0.2),        # Increased contrast
                layers.RandomBrightness(0.15),     # Increased brightness
                layers.RandomTranslation(0.15, 0.15),  # Increased translation
                # Add noise for more regularization
                layers.GaussianNoise(0.01),
            ])
            dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y),
                                num_parallel_calls=tf.data.AUTOTUNE)
        
        # Normalize to [0,1] range
        dataset = dataset.map(lambda x, y: (tf.cast(x, tf.float32)/255.0, y),
                            num_parallel_calls=tf.data.AUTOTUNE)
        
        # Cache and prefetch for performance
        dataset = dataset.cache().prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def build_model(self):
        """Build heavily regularized lightweight CNN to prevent overfitting"""
        self.model = keras.Sequential([
            layers.Input(shape=(self.img_height, self.img_width, 3)),
            
            # First conv block - very small and heavily regularized
            layers.Conv2D(6, (3,3), activation='relu', padding='same', 
                         kernel_regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),  # Early dropout
            layers.MaxPooling2D((2,2)),
            
            # Second conv block - small with heavy regularization
            layers.Conv2D(12, (3,3), activation='relu', padding='same',
                         kernel_regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.25),
            layers.MaxPooling2D((2,2)),
            
            # Third conv block - minimal complexity
            layers.Conv2D(16, (3,3), activation='relu', padding='same',
                         kernel_regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.MaxPooling2D((2,2)),
            
            # Classifier with extreme regularization
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.6),   # Very high dropout
            layers.Dense(8, activation='relu',  # Very small dense layer
                        kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])
        
        # Much lower learning rate for stable learning
        optimizer = keras.optimizers.Adam(learning_rate=0.0005, decay=1e-5)
        
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        print("Heavily Regularized Model Architecture:")
        self.model.summary()
        
        # Calculate model parameters
        total_params = self.model.count_params()
        print(f"\nTotal parameters: {total_params:,}")
        print("Anti-overfitting model is ready for training!")
        
        return self.model
    
    def train_model(self, train_ds, val_ds, epochs=50):  # More epochs with slower learning
        """Train with extreme regularization to prevent overfitting"""
        
        # Very aggressive callbacks to prevent overfitting
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',  # Monitor val_loss instead of val_accuracy
            patience=12,         # Increased patience
            restore_best_weights=True,
            mode='min',
            min_delta=0.005     # Larger minimum delta
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,      # More aggressive LR reduction
            patience=5,      # Faster LR reduction
            min_lr=1e-9    # Lower minimum LR
        )
        
        # Model checkpoint to save best model based on validation loss
        checkpoint = keras.callbacks.ModelCheckpoint(
            'best_microplastic_model_antioverfit.keras',
            monitor='val_loss',  # Save based on val_loss
            save_best_only=True,
            mode='min'
        )
        
        # Custom callback to monitor overfitting
        class OverfitMonitor(keras.callbacks.Callback):
            def __init__(self):
                self.best_val_loss = float('inf')
                self.patience_counter = 0
                
            def on_epoch_end(self, epoch, logs=None):
                train_acc = logs.get('accuracy', 0)
                val_acc = logs.get('val_accuracy', 0)
                train_loss = logs.get('loss', 0)
                val_loss = logs.get('val_loss', 0)
                
                # Calculate gaps
                acc_gap = train_acc - val_acc
                loss_gap = val_loss - train_loss
                
                if epoch > 5:  # Start monitoring after 5 epochs
                    if acc_gap > 0.15 or loss_gap > 0.3:
                        print(f"\n⚠ OVERFITTING DETECTED at epoch {epoch+1}:")
                        print(f"   Accuracy gap: {acc_gap:.3f}, Loss gap: {loss_gap:.3f}")
                        
        overfit_monitor = OverfitMonitor()
        
        # Exponential decay scheduler
        def exp_decay(epoch):
            initial_lrate = 0.0005
            k = 0.05
            lrate = initial_lrate * np.exp(-k*epoch)
            return max(lrate, 1e-8)
        
        lr_scheduler = keras.callbacks.LearningRateScheduler(exp_decay)
        
        print("Starting anti-overfitting training...")
        print("Focus: Stable validation performance over training accuracy")
        
        history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=[early_stopping, reduce_lr, checkpoint, lr_scheduler, overfit_monitor]
        )
        
        return history
    
    def evaluate_model(self, test_ds):
        """Evaluate model and show detailed metrics"""
        print("Evaluating anti-overfitting model on test set...")
        results = self.model.evaluate(test_ds)
        test_loss, test_accuracy = results[0], results[1]
        
        if len(results) > 2:
            test_precision, test_recall = results[2], results[3]
            f1_score = 2 * (test_precision * test_recall) / (test_precision + test_recall + 1e-7)
            print(f"\nDetailed Test Results:")
            print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
            print(f"Test Precision: {test_precision:.4f}")
            print(f"Test Recall: {test_recall:.4f}")
            print(f"Test F1-Score: {f1_score:.4f}")
        else:
            print(f"\nTest Results:")
            print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        
        print(f"Test Loss: {test_loss:.4f}")
        
        # Get detailed predictions
        y_pred, y_true = [], []
        prediction_probs = []
        
        for images, labels in test_ds:
            predictions = self.model.predict(images)
            prediction_probs.extend(predictions.flatten())
            y_pred.extend((predictions > 0.5).astype(int).flatten())
            y_true.extend(labels.numpy())
        
        # Classification report
        class_names = ['Clean_Water', 'Microplastics']
        print(f"\nDetailed Classification Report:")
        print(classification_report(y_true, y_pred, target_names=class_names))
        
        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm := confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix - Anti-Overfitting Microplastic Detection")
        plt.tight_layout()
        plt.show()
        
        # Show sample predictions with confidence scores
        print(f"\nSample Predictions:")
        for i in range(min(8, len(y_true))):
            true_label = class_names[y_true[i]]
            pred_label = class_names[y_pred[i]]
            confidence = prediction_probs[i] if y_pred[i] == 1 else (1 - prediction_probs[i])
            status = "CORRECT" if y_true[i] == y_pred[i] else "WRONG"
            print(f"  {status} Sample {i+1}: True={true_label}, Predicted={pred_label}, Confidence={confidence:.3f}")
        
        return test_accuracy
    
    def convert_to_tflite(self, model_name="microplastic_cnn_antioverfit"):
        """Convert model to TensorFlow Lite (ESP32-S3 optimized)"""
        print("Converting anti-overfitting model to TensorFlow Lite...")
        
        # Standard conversion with optimizations
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        converter.experimental_new_converter = True
        
        # Convert model
        tflite_model = converter.convert()
        
        # Save TFLite model
        tflite_filename = f"{model_name}.tflite"
        with open(tflite_filename, 'wb') as f:
            f.write(tflite_model)
        
        print(f"TFLite model saved as: {tflite_filename}")
        print(f"Model size: {len(tflite_model)/1024:.2f} KB")
        
        # Verify the model works
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
        """Plot training curves with overfitting analysis"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Anti-Overfitting Training Analysis - Microplastic Detection', fontsize=16)
        
        epochs = range(1, len(history.history['accuracy']) + 1)
        
        # Plot accuracy
        axes[0,0].plot(epochs, history.history['accuracy'], 'bo-', label='Training Accuracy', linewidth=2)
        axes[0,0].plot(epochs, history.history['val_accuracy'], 'ro-', label='Validation Accuracy', linewidth=2)
        axes[0,0].set_title('Model Accuracy (Anti-Overfitting)')
        axes[0,0].set_xlabel('Epoch')
        axes[0,0].set_ylabel('Accuracy')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Plot loss
        axes[0,1].plot(epochs, history.history['loss'], 'bo-', label='Training Loss', linewidth=2)
        axes[0,1].plot(epochs, history.history['val_loss'], 'ro-', label='Validation Loss', linewidth=2)
        axes[0,1].set_title('Model Loss (Anti-Overfitting)')
        axes[0,1].set_xlabel('Epoch')
        axes[0,1].set_ylabel('Loss')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot accuracy gap over time
        acc_gap = [train - val for train, val in zip(history.history['accuracy'], history.history['val_accuracy'])]
        axes[0,2].plot(epochs, acc_gap, 'go-', label='Train-Val Accuracy Gap', linewidth=2)
        axes[0,2].axhline(y=0.05, color='r', linestyle='--', label='Healthy Gap (5%)')
        axes[0,2].axhline(y=0.10, color='orange', linestyle='--', label='Warning Gap (10%)')
        axes[0,2].set_title('Overfitting Monitor (Accuracy Gap)')
        axes[0,2].set_xlabel('Epoch')
        axes[0,2].set_ylabel('Accuracy Gap')
        axes[0,2].legend()
        axes[0,2].grid(True, alpha=0.3)
        
        # Plot precision
        if 'precision' in history.history:
            axes[1,0].plot(epochs, history.history['precision'], 'go-', label='Training Precision', linewidth=2)
            axes[1,0].plot(epochs, history.history['val_precision'], 'mo-', label='Validation Precision', linewidth=2)
            axes[1,0].set_title('Model Precision')
            axes[1,0].set_xlabel('Epoch')
            axes[1,0].set_ylabel('Precision')
            axes[1,0].legend()
            axes[1,0].grid(True, alpha=0.3)
        
        # Plot recall
        if 'recall' in history.history:
            axes[1,1].plot(epochs, history.history['recall'], 'co-', label='Training Recall', linewidth=2)
            axes[1,1].plot(epochs, history.history['val_recall'], 'yo-', label='Validation Recall', linewidth=2)
            axes[1,1].set_title('Model Recall')
            axes[1,1].set_xlabel('Epoch')
            axes[1,1].set_ylabel('Recall')
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)
        
        # Plot learning rate over time
        if hasattr(self, 'lr_history'):
            axes[1,2].plot(epochs, self.lr_history, 'po-', label='Learning Rate', linewidth=2)
            axes[1,2].set_title('Learning Rate Schedule')
            axes[1,2].set_xlabel('Epoch')
            axes[1,2].set_ylabel('Learning Rate')
            axes[1,2].set_yscale('log')
            axes[1,2].legend()
            axes[1,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Comprehensive training analysis
        best_val_acc = max(history.history['val_accuracy'])
        best_val_acc_epoch = history.history['val_accuracy'].index(best_val_acc) + 1
        final_val_acc = history.history['val_accuracy'][-1]
        final_train_acc = history.history['accuracy'][-1]
        
        best_val_loss = min(history.history['val_loss'])
        best_val_loss_epoch = history.history['val_loss'].index(best_val_loss) + 1
        final_val_loss = history.history['val_loss'][-1]
        final_train_loss = history.history['loss'][-1]
        
        # Calculate gaps
        final_acc_gap = final_train_acc - final_val_acc
        final_loss_gap = final_val_loss - final_train_loss
        
        print(f"\n" + "="*60)
        print(f"ANTI-OVERFITTING TRAINING ANALYSIS")
        print(f"="*60)
        print(f"Best validation accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%) at epoch {best_val_acc_epoch}")
        print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_val_loss_epoch}")
        print(f"\nFinal Metrics:")
        print(f"  Training accuracy: {final_train_acc:.4f} ({final_train_acc*100:.2f}%)")
        print(f"  Validation accuracy: {final_val_acc:.4f} ({final_val_acc*100:.2f}%)")
        print(f"  Training loss: {final_train_loss:.4f}")
        print(f"  Validation loss: {final_val_loss:.4f}")
        print(f"\nOverfitting Assessment:")
        print(f"  Accuracy gap: {final_acc_gap:.4f} ({final_acc_gap*100:.2f}%)")
        print(f"  Loss gap: {final_loss_gap:.4f}")
        
        # Overfitting diagnosis
        if final_acc_gap < 0.03 and final_loss_gap < 0.2:
            print("   EXCELLENT: Minimal overfitting detected!")
            status = "EXCELLENT"
        elif final_acc_gap < 0.08 and final_loss_gap < 0.4:
            print("   GOOD: Well-controlled overfitting!")
            status = "GOOD"
        elif final_acc_gap < 0.15 and final_loss_gap < 0.6:
            print("    ACCEPTABLE: Some overfitting present but manageable!")
            status = "ACCEPTABLE"
        else:
            print("   POOR: Significant overfitting detected!")
            status = "NEEDS_IMPROVEMENT"
            
        print(f"  Total epochs: {len(epochs)}")
        print(f"  Model status: {status}")
        print(f"="*60)
        
        return status

# ---------------- Main Execution ----------------
def main():
    """Anti-overfitting training pipeline"""
    print("Anti-Overfitting Microplastic Detection - TinyML Training Pipeline")
    print("=" * 80)
    print("Target: ESP32-S3 Compatible Model with Minimal Overfitting")
    print("=" * 80)
    
    # Initialize classifier with anti-overfitting settings
    classifier = MicroplasticImageClassifier(img_height=64, img_width=64, batch_size=32)
    
    # UPDATE THIS PATH to your dataset folder
    data_directory = "E:/ml/microplastic/dataset"
    
    try:
        # Load datasets
        print("Step 1: Loading datasets...")
        train_ds, val_ds, test_ds = classifier.load_dataset(data_directory)
        
        # Preprocess datasets with heavy augmentation
        print("\nStep 2: Preprocessing with heavy augmentation...")
        train_ds = classifier.preprocess_dataset(train_ds, augment=True)
        val_ds = classifier.preprocess_dataset(val_ds, augment=False)
        test_ds = classifier.preprocess_dataset(test_ds, augment=False)
        print("Heavy augmentation applied!")
        
        # Build anti-overfitting model
        print("\nStep 3: Building anti-overfitting CNN model...")
        classifier.build_model()
        
        # Train with anti-overfitting strategy
        print("\nStep 4: Training with anti-overfitting strategy...")
        history = classifier.train_model(train_ds, val_ds, epochs=50)
        
        # Evaluate model
        print("\nStep 5: Evaluating anti-overfitting model...")
        test_accuracy = classifier.evaluate_model(test_ds)
        
        # Convert to TensorFlow Lite
        print("\nStep 6: Converting to TensorFlow Lite for ESP32-S3...")
        tflite_file = classifier.convert_to_tflite("microplastic_image_classifier_antioverfit")
        
        # Comprehensive analysis
        print("\nStep 7: Comprehensive training analysis...")
        training_status = classifier.plot_training_history(history)
        
       
        
        print(f"\nReady for ESP32-S3 deployment: {tflite_file}")
        print("=" * 80)
            
    except FileNotFoundError as e:
        print(f"Dataset Error: {e}")
        print("Please check your dataset path and folder structure.")
    except Exception as e:
        print(f"Training Error: {e}")
        print("Please check your environment setup and try again.")

if __name__ == "__main__":
    main()