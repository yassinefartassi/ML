import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

# Set image parameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32

def create_model():
    """Create a CNN model for image classification"""
    model = models.Sequential([
        layers.Conv2D(32, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Binary classification (good vs bad quality)
    ])
    return model

def prepare_data():
    """Prepare and augment the training data"""
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    # Load training data
    train_generator = train_datagen.flow_from_directory(
        'dataset',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training'
    )

    # Load validation data
    validation_generator = train_datagen.flow_from_directory(
        'dataset',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation'
    )

    return train_generator, validation_generator

def train_model():
    """Train the model on the prepared data"""
    # Create and compile the model
    model = create_model()
    model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy'])

    # Prepare the data
    train_generator, validation_generator = prepare_data()

    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=15,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE
    )

    # Save the model
    model.save('quality_detection_model.h5')
    return history, model

def plot_training_results(history):
    """Plot the training and validation accuracy/loss"""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.close()

def predict_quality(model, image_path):
    """Predict the quality of a single image"""
    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(IMG_HEIGHT, IMG_WIDTH)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)
    return "Good Quality" if prediction[0] > 0.5 else "Bad Quality"

if __name__ == "__main__":
    # Check if we have enough images
    good_quality_path = os.path.join('dataset', 'good_quality')
    bad_quality_path = os.path.join('dataset', 'bad_quality')
    
    good_images = len([f for f in os.listdir(good_quality_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
    bad_images = len([f for f in os.listdir(bad_quality_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
    
    if good_images == 0 or bad_images == 0:
        print("Please add images to both 'good_quality' and 'bad_quality' folders before training!")
        print(f"Current images count - Good Quality: {good_images}, Bad Quality: {bad_images}")
    else:
        print("Starting model training...")
        history, model = train_model()
        plot_training_results(history)
        print("\nModel training completed!")
        print("Model has been saved as 'quality_detection_model.h5'")
        print("Training results plot has been saved as 'training_results.png'")
