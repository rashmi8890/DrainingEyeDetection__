# DrainingEyeDetection__
This project trains a Convolutional Neural Network (CNN) to detect drowsiness from grayscale images using TensorFlow and Keras. The dataset is loaded using ImageDataGenerator, and the model is trained to classify images into two categories.

# Dataset Preparation
Save Dataset in following formate:
- archive
  - data
    - train
      - open eyes
      - close eyes
    - test
      - open eyes
      - close eyes

# Dependencies
Make sure you install **tensorflow**, **openCV**, **pygame** and **numpy**
``` python
pip install tensorflow opencv-python pygame numpy
```
# Data Preprosessing
Preprocess the Data is important before training the model. Here we generate many randome image by using image of dataset. So that model can train on different random images.
 ``` python
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)
```
# Training the model
After doing the preprocessing, next step is to train model on given dataset. So firstly we divide data in trainig data and validation data.
- Conv2D layers with ReLU activation for feature extraction
- MaxPooling2D layers for dimensionality reduction
- Flatten layer to convert feature maps into a 1D vector
- Dense layers with ReLU and Softmax activation

```python
train_data = datagen.flow_from_directory(
    r"C:\Users\dell\Downloads\archive (2)\data\train",
    target_size=IMG_SIZE,
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=BATCH,
    subset='training',
)

val_data = datagen.flow_from_directory(
    r"C:\Users\dell\Downloads\archive (2)\data\train",
    target_size=IMG_SIZE,
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=BATCH,
    subset='validation',
)
```
# Model Architecture
Model consist three layer of convolution with maxpooling, dense, flattening layer
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling2D, Conv2D, Dense, Flatten, Dropout

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(24,24,1)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_data, validation_data=val_data, epochs=20)

model.save("drowsiness_model.keras")
```
# Save Model
```python
model.save("drowsiness_model.keras")
```

# Features

- Loads a pre-trained deep learning model (drowsiness_model.keras).
- Detects eyes using OpenCV's Haar cascades.
- Predicts whether eyes are open or closed using a CNN.
- Triggers an alert sound when drowsiness is detected.

# Model Loading and Testing
```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model1 = tf.keras.models.load_model("drowsiness_model.keras")

test_datagen = ImageDataGenerator(rescale=1./255)

test_data = test_datagen.flow_from_directory(
    r"C:\Users\dell\Downloads\archive (2)\data\test",
    target_size=(24, 24),
    color_mode='grayscale',
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

predictions = model1.predict(test_data)
predicted_labels = predictions.argmax(axis=1)
```
# Evaluating Accuracy
``` python
from sklearn.metrics import accuracy_score
true_labels = test_data.classes
accuracy = accuracy_score(true_labels, predicted_labels)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
```
# Single image prediction
```python
from tensorflow.keras.preprocessing import image
img_path = r"C:\Users\dell\Downloads\archive (2)\data\test\open eyes\s0005_00719_0_0_1_0_0_01.png"
img = image.load_img(img_path, target_size=(24,24), color_mode='grayscale')
img_array = image.img_to_array(img)/255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model1.predict(img_array)
label = "Open" if prediction.argmax() == 1 else "Closed"
print(f"Prediction: {label}")
```
# Real time Drowsiness detection
The program uses OpenCV to detect faces and eyes in a live video stream.
```python
import cv2, pygame, threading
pygame.mixer.init()

def play_alert():
    sound = pygame.mixer.Sound(r"C:\Users\dell\Downloads\alert-109578.mp3")
    sound.play()

cap = cv2.VideoCapture(0)
drowsy_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(face_roi)

        for (ex, ey, ew, eh) in eyes:
            eye_roi = face_roi[ey:ey+eh, ex:ex+ew]
            eye_roi = cv2.resize(eye_roi, (24, 24)) / 255.0
            eye_roi = np.expand_dims(eye_roi, axis=(0, -1))

            prediction = model1.predict(eye_roi)
            label = "Open" if prediction.argmax() == 1 else "Closed"

            if label == "Closed":
                drowsy_counter += 1
            else:
                drowsy_counter = 0

            if drowsy_counter > 15:
                threading.Thread(target=play_alert, daemon=True).start()
                cv2.putText(frame, "⚠️ DROWSINESS ALERT!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

# Future Improvement
- Improve model accuracy with more training data.
- Implement mobile app integration.
- Enhance real-time performance optimization
