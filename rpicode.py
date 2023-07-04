
#### code for Raspberry pi for detection of cow and human
# This code is written by AMAN VERMA
# importing libraries
import RPi.GPIO as GPIO
import time
import picamera
import numpy as np
import tensorflow as tf
import cv2
import gsm_module

# Setting up the pins for PIR sensor and buzzer
pir_pin = 14
buzzer_pin = 18
GPIO.setmode(GPIO.BCM)
GPIO.setup(pir_pin, GPIO.IN)
GPIO.setup(buzzer_pin, GPIO.OUT)

# Loading the trained model
model = tf.keras.models.load_model("tflite_model")

# Preprocessing function
def preprocess_image(img):
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# GSM module initialization
gsm = gsm_module.GSM_Module()

# Main loop
while True:
    # Waiting for motion detection
    while GPIO.input(pir_pin) == 0:
        time.sleep(0.1)
    print("Motion detected!")

    # Capturing the image from the camera
    with picamera.PiCamera() as camera:
        camera.resolution = (1024, 768)
        camera.start_preview()
        time.sleep(2)
        camera.capture("image.jpg")

    # Preprocessing the image
    img = cv2.imread("image.jpg")
    img = preprocess_image(img)

    # Making prediction using the loaded model
    pred = model.predict(img)
    label = np.argmax(pred)

    # Handling the prediction result
    if label == 0:  # Cow detected
        print("Dangerous!")
        GPIO.output(buzzer_pin, GPIO.HIGH)
        gsm.send_sms("Dangerous!")
    elif label == 1:  # People detected
        print("Not dangerous.")
        while GPIO.input(pir_pin) == 1:
            time.sleep(0.1)
    else:  # Unknown object detected
        print("Not sure.")
        while GPIO.input(pir_pin) == 1:
            time.sleep(0.1)

    # Resetting the buzzer
    GPIO.output(buzzer_pin, GPIO.LOW)