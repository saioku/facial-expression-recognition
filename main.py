import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, regularizers
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import seaborn as sns

# ===========================================
# Data Preprocessing and Transformation
# ===========================================
def preprocess_data():
    image_folder_path = './yalefaces'
    output_folder_path = './preprocessed_expressions'

    if not os.path.isdir(output_folder_path):
        os.mkdir(output_folder_path)

    for file in os.listdir(image_folder_path):
        # Filter out non-image files and images with lighting variations and glasses
        if 'gif' not in file and 'txt' not in file and 'DS_Store' not in file:
            expression = file.split('.')[1]
            if 'light' in expression or 'glasses' in expression:
                continue  # Skip lighting variations and images with glasses
            
            img = Image.open(f'{image_folder_path}/{file}').convert('L')
            img = img.resize((48, 48))  # Resize for consistency
            expression_folder = f'{output_folder_path}/{expression}'
            if not os.path.isdir(expression_folder):
                os.mkdir(expression_folder)
            output_img_path = f'{expression_folder}/{file}.jpg'
            img.save(output_img_path)

# ===========================================
# Load Data and Prepare for Training
# ===========================================
def load_data(data_dir):
    # image augmentation
    data_gen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    train_data = data_gen.flow_from_directory(
        data_dir, target_size=(48, 48), color_mode="grayscale", batch_size=32, class_mode="categorical", subset="training")
    val_data = data_gen.flow_from_directory(
        data_dir, target_size=(48, 48), color_mode="grayscale", batch_size=32, class_mode="categorical", subset="validation")
    return train_data, val_data

# ===========================================
# Define Model
# ===========================================
def build_model(input_shape=(48, 48, 1), num_classes=5):
    # L2 regularization
    l2_reg = regularizers.l2(0.001)

    model = models.Sequential()

    # Block 1
    model.add(layers.Conv2D(32, (3, 3), padding='same', input_shape=input_shape, kernel_regularizer=l2_reg))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2_reg))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # Block 2
    model.add(layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2_reg))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2_reg))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # Block 3
    model.add(layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2_reg))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2_reg))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Flatten())

    # fully connected Layers
    model.add(layers.Dense(256, kernel_regularizer=l2_reg))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(num_classes, activation='softmax', kernel_regularizer=l2_reg))

    # learning rate and label smoothing
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
                  metrics=['accuracy'])
    return model

# ===========================================
# Custom Callback to Show Epoch Progress (unchanged)
# ===========================================
class CustomEpochCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {logs['loss']:.4f}")

    def on_train_end(self, logs=None):
        final_accuracy = logs['accuracy'] * 100
        print(f"\nFinal Model Accuracy: {final_accuracy:.2f}%")

# ===========================================
# Training and Evaluation
# ===========================================
def train_and_evaluate(model, train_data, val_data):
    custom_callback = CustomEpochCallback()        
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=1000,
        verbose=0,
        callbacks=[custom_callback]
    )
    return history

# ===========================================
# Display Results
# ===========================================
def display_results(model, data_gen):
    class_labels = list(data_gen.class_indices.keys())
    X, y_true, y_pred, filenames = [], [], [], []
    
    data_gen.reset()
    for images, labels in data_gen:
        preds = model.predict(images)
        y_true.extend(np.argmax(labels, axis=1))
        y_pred.extend(np.argmax(preds, axis=1))
        X.extend(images)
        filenames.extend(data_gen.filenames)
        if len(y_true) >= data_gen.samples:
            break
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # classification report and F1 score
    report = classification_report(y_true, y_pred, target_names=class_labels)
    print(report)
    print("F1 Score:", f1_score(y_true, y_pred, average='weighted'))

    # display a few correct and incorrect predictions
    correct_idx = np.where(y_true == y_pred)[0][:5]
    incorrect_idx = np.where(y_true != y_pred)[0][:5]

    def plot_images(indices, title):
        plt.figure(figsize=(10, 5))
        plt.suptitle(title)
        for i, idx in enumerate(indices, 1):
            plt.subplot(1, 5, i)
            plt.imshow(X[idx].reshape(48, 48), cmap='gray')
            plt.title(f"Pred: {class_labels[y_pred[idx]]}\nTrue: {class_labels[y_true[idx]]}")
            plt.axis('off')
        plt.show()

    if len(correct_idx) > 0:
        plot_images(correct_idx, "Correct Predictions")
    if len(incorrect_idx) > 0:
        plot_images(incorrect_idx, "Incorrect Predictions")

# ===========================================
# Plot Accuracy and Loss Curves
# ===========================================
def plot_training_curves(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history.get('val_accuracy', []), label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history.get('val_loss', []), label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

# ===========================================
# Main Script
# ===========================================
if __name__ == "__main__":
    # preprocess data if not done already
    preprocess_data()

    # load preprocessed data
    data_dir = './preprocessed_expressions'
    train_data, val_data = load_data(data_dir)

    # build and train model
    model = build_model(num_classes=train_data.num_classes)
    history = train_and_evaluate(model, train_data, val_data)

    # plot training curves
    plot_training_curves(history)

    # rvaluate and display results
    display_results(model, val_data)
