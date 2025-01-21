import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import Xception
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the paths to the training and test data
train_dir = "ML_Dataset/RGB-M/train"
test_dir = "ML_Dataset/RGB-M/test"

# Image data generator for training
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values to [0, 1]
    rotation_range=30,  # Increased range for more variability
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest')

# Image data generator for testing (only rescaling)
test_datagen = ImageDataGenerator(rescale=1./255)

# Flow from directory for training and testing data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),  # Resize to match the input shape (Xception input size is 224x224)
    batch_size=32,
    class_mode='sparse')  # sparse_categorical_crossentropy expects integer labels

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),  # Resize to match the input shape
    batch_size=32,
    class_mode='sparse')

# Load the pre-trained Xception model
base_model = Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Unfreeze the last few layers of the base model for fine-tuning
base_model.trainable = True
fine_tune_at = 100  # Fine-tune layers after this layer
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Build the model with a custom classification head
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    BatchNormalization(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.4),
    Dense(4, activation='softmax')  # 4 classes (seasons)
])

# Compile the model with an improved optimizer
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Define learning rate scheduler and early stopping callback
lr_scheduler = ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model with increased epochs to allow fine-tuning
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=test_generator,
    validation_steps=test_generator.samples // test_generator.batch_size,
    epochs=50,  # Can increase epochs for better training
    callbacks=[lr_scheduler, early_stopping]
)

test_loss, test_acc = model.evaluate(test_generator)
print(f"Test accuracy: {test_acc * 100:.2f}%")
