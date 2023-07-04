
# This code trains the CNN model and convert it into tflite model,
# So it can run on limited resource based hardware systems like raspberrypi 3 B+
# This code is written by AMAN VERMA
# Importing the necessary libraries
import tensorflow as tf
import numpy as np
import os
import cv2

# Connecting the drive to this file
from google.colab import drive
drive.mount('/content/drive')

# Defining the paths to the dataset directories
cow_path = "/content/drive/MyDrive/dataset/cow"
people_path = "/content/drive/MyDrive/dataset/people"

# Defining the input image dimensions
img_height = 224
img_width = 224
num_channels = 3

# Creating a function to load the dataset
def load_data(data_dir):
    # Reading the image files and resizing them
    data = []
    for img in os.listdir(data_dir):
        img_path = os.path.join(data_dir, img)
        img_arr = cv2.imread(img_path)
        resized_arr = cv2.resize(img_arr, (img_height, img_width))
        data.append(resized_arr)

    return np.array(data)

# Loading the datasets
cow_data = load_data(cow_path)
people_data = load_data(people_path)

# Defining the labels
cow_labels = np.zeros(len(cow_data))
people_labels = np.ones(len(people_data))
#bluebuck_labels = np.full(len(bluebuck_data), 2)

# Combining the data and labels
data = np.concatenate((cow_data, people_data), axis=0)
labels = np.concatenate((cow_labels, people_labels), axis=0)

# Splitting the data into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Normalizing the pixel values
X_train = X_train / 255.0
X_test = X_test / 255.0

# Defining the model architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(img_height, img_width, num_channels)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

# Print the summary of the model architecture
model.summary()

# Compiling the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Training the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Evaluating the model performance
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test accuracy:", test_acc)

# Saving the model as a .h5 file
model.save("my_model.h5")

# Converting the model to a tflite model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Saving the tflite model to a file
with open("my_model.tflite", 'wb') as f:
    f.write(tflite_model)