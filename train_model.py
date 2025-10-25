import os
import json
import numpy as np
try:
    import tensorflow as tf
except Exception as e:
    raise ImportError("TensorFlow import failed. Install tensorflow (e.g. pip install tensorflow).") from e


ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator
Sequential = tf.keras.models.Sequential
Conv2D = tf.keras.layers.Conv2D
MaxPooling2D = tf.keras.layers.MaxPooling2D
BatchNormalization = tf.keras.layers.BatchNormalization
Dropout = tf.keras.layers.Dropout
Flatten = tf.keras.layers.Flatten
Dense = tf.keras.layers.Dense
ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint
ReduceLROnPlateau = tf.keras.callbacks.ReduceLROnPlateau
EarlyStopping = tf.keras.callbacks.EarlyStopping

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, 'fer2013')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'test')


IMG_SIZE = (64, 64)   
BATCH_SIZE = 64
EPOCHS = 40
LABEL_SMOOTHING = 0.05  

train_gen = ImageDataGenerator(rescale=1./255,
                               rotation_range=12,
                               width_shift_range=0.12,
                               height_shift_range=0.12,
                               shear_range=0.12,
                               zoom_range=0.12,
                               horizontal_flip=True)
val_gen = ImageDataGenerator(rescale=1./255)

train_generator = train_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

validation_generator = val_gen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

num_classes = len(train_generator.class_indices)

# Simple CNN 
def build_model(input_shape=(64,64,1), num_classes=7):
    model = Sequential()
    model.add(Conv2D(64, (3,3), activation='relu', padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING)
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    return model

model = build_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1), num_classes=num_classes)

checkpoint_path = os.path.join(BASE_DIR, 'emotion_model.h5')
callbacks = [
    ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1),
    EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
]


counts = np.bincount(train_generator.classes)
total = counts.sum()
class_weight = {}
for i, c in enumerate(counts):
    if c > 0:
        class_weight[i] = total / (len(counts) * c)
    else:
        class_weight[i] = 1.0

steps_per_epoch = max(1, train_generator.samples // BATCH_SIZE)
validation_steps = max(1, validation_generator.samples // BATCH_SIZE)

model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=callbacks,
    class_weight=class_weight   
)

labels = {v: k for k, v in train_generator.class_indices.items()}
with open(os.path.join(BASE_DIR, 'labels.json'), 'w') as f:
    json.dump(labels, f)
print("Training finished. Model saved to:", checkpoint_path)
