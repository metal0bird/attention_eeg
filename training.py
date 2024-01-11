import os
import zipfile
import random
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile
import matplotlib.pyplot as plt
from keras.models import load_model 


FOCUS_SOURCE_DIR = "dataset/focus/"
UNFOCUS_SOURCE_DIR = "dataset/unfocus/"
DROWSY_SOURCE_DIR = "dataset/drowsy/"

TRAINING_DIR = 'images/training/'
VALIDATION_DIR = 'images/validation/'

TRAINING_FOCUS_DIR = os.path.join(TRAINING_DIR, "focus/")
VALIDATION_FOCUS_DIR = os.path.join(VALIDATION_DIR, "focus/")

TRAINING_UNFOCUS_DIR = os.path.join(TRAINING_DIR, "unfocus/")
VALIDATION_UNFOCUS_DIR = os.path.join(VALIDATION_DIR, "unfocus/")

TRAINING_DROWSY_DIR = os.path.join(TRAINING_DIR, "drowsy/")
VALIDATION_DROWSY_DIR = os.path.join(VALIDATION_DIR, "drowsy/")


with tf.device('/GPU:0'):
    model = tf.keras.models.Sequential([
        # Note the input shape is the desired size of the image 150x150 with 3 bytes color
        # This is the first convolution
        tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The second convolution
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        # The third convolution
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        # The fourth convolution
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        # Flatten the results to feed into a DNN
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        # 512 neuron hidden layer
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    # Print the model summary
    #model.summary()

    # Set the training parameters
    model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    #if it doesn't work use loss='sparse_categorical_crossentropy'


    def train_val_generators(TRAINING_DIR, VALIDATION_DIR):

        train_datagen = ImageDataGenerator(rescale=1.0/255.0)

        # Pass in the appropriate arguments to the flow_from_directory method
        train_generator = train_datagen.flow_from_directory(directory=TRAINING_DIR,
                                                            batch_size=30,
                                                            class_mode='categorical',
                                                            target_size=(150, 150))

        # Instantiate the ImageDataGenerator class (don't forget to set the rescale argument)
        validation_datagen = ImageDataGenerator(rescale=1.0/255.0)

        # Pass in the appropriate arguments to the flow_from_directory method
        validation_generator = validation_datagen.flow_from_directory(directory=VALIDATION_DIR,
                                                                        batch_size=30,
                                                                        class_mode='categorical',
                                                                        target_size=(150, 150))
        ### END CODE HERE
        return train_generator, validation_generator

    train_generator, validation_generator = train_val_generators(TRAINING_DIR, VALIDATION_DIR)

    # Train the model.
    history = model.fit(
                train_generator,
                validation_data = validation_generator,
                steps_per_epoch = 100,
                epochs = 40,
                validation_steps = 50,
                verbose = 2)

    model.save("network.h5") 
#loaded_model = load_model("network.h5") 
#loss, accuracy = loaded_model.evaluate(test_data, test_targets) 