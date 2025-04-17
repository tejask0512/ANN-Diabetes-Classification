# model_training.py
# This file contains the data preprocessing, model creation, training, and evaluation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import datetime
import os

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load the dataset
print("Loading and preprocessing data...")
df = pd.read_csv("data/diabetes.csv")

# Display basic information
print("Dataset Information:")
print(df.info())
print("\nStatistical Summary:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Exploratory Data Analysis
print("\nCreating exploratory visualizations...")

# Correlation matrix
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.savefig('visualizations/correlation_matrix.png')
plt.close()

# Feature distributions
plt.figure(figsize=(15, 10))
for i, column in enumerate(df.columns):
    plt.subplot(3, 3, i+1)
    sns.histplot(df[column], kde=True)
    plt.title(f'Distribution of {column}')
plt.tight_layout()
plt.savefig('visualizations/feature_distributions.png')
plt.close()

# Class distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='Outcome', data=df)
plt.title('Class Distribution')
plt.savefig('visualizations/class_distribution.png')
plt.close()

# Separate features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler for later use in the app
import pickle
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Define the model architecture
def create_model():
    model = keras.Sequential([
        keras.layers.Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Create the model
print("\nCreating and compiling model...")
model = create_model()
model.summary()

# Set up TensorBoard callback
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
    write_graph=True
)

# Early stopping to prevent overfitting
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Model checkpoint to save the best model
os.makedirs('models', exist_ok=True)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'models/best_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)

# Train the model
print("\nTraining the model...")
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[tensorboard_callback, early_stopping, model_checkpoint]
)

# Plot training history
plt.figure(figsize=(12, 5))

# Plot training & validation accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')

# Plot training & validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

plt.tight_layout()
plt.savefig('visualizations/training_history.png')
plt.close()

# Load the best model
best_model = keras.models.load_model('models/best_model.h5')

# Evaluate the model on the test set
print("\nEvaluating the model on test data...")
test_loss, test_accuracy = best_model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Make predictions
y_pred_prob = best_model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

# Calculate evaluation metrics
print("\nGenerating evaluation metrics...")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('visualizations/confusion_matrix.png')
plt.close()

# Classification report
report = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(report)

# Save the classification report to a file
with open('evaluation/classification_report.txt', 'w') as f:
    f.write(report)

# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.savefig('visualizations/roc_curve.png')
plt.close()

print("\nTraining and evaluation complete!")
print(f"TensorBoard logs saved to: {log_dir}")
print("Best model saved to: models/best_model.h5")
print("Visualizations saved to: visualizations/")

# Print instructions for viewing TensorBoard
print("\nTo view training metrics in TensorBoard, run:")
print(f"tensorboard --logdir={os.path.abspath(log_dir)}")