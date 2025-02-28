import numpy as np
import tensorflow as tf
import cv2
import pygame
pygame.mixer.init()
import threading
from tensorflow.keras.preprocessing import image

model1 = tf.keras.models.load_model("drowsiness_model.keras")

from tensorflow.keras.preprocessing.image import ImageDataGenerator

test_datagen = ImageDataGenerator(rescale=1./255)

test_data = test_datagen.flow_from_directory(
    r"C:\Users\dell\Downloads\archive (2)\data\test",
    target_size=(24, 24),
    color_mode='grayscale',
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # To keep filenames in order
)

# Get predictions for all test images
predictions = model1.predict(test_data)

# Get predicted labels
predicted_labels = predictions.argmax(axis=1)
print(predicted_labels)

from sklearn.metrics import accuracy_score
# Get actual labels
true_labels = test_data.classes

# Calculate accuracy
accuracy = accuracy_score(true_labels, predicted_labels)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

img_path=r"C:\Users\dell\Downloads\archive (2)\data\test\open eyes\s0005_00719_0_0_1_0_0_01.png"
img=image.load_img(img_path, target_size=(24,24), color_mode='grayscale')
img_array=image.img_to_array(img)/255.0
img_array=np.expand_dims(img_array,axis=0)

prediction=model1.predict(img_array)
label = "Open" if prediction.argmax()==1 else "Closed"
print(f"Prediction: {label}")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

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

            # Predict
            prediction = model1.predict(eye_roi)
            label = "Open" if prediction.argmax() == 1 else "Closed"

            # Track drowsiness
            if label == "Closed":
                drowsy_counter += 1
            else:
                drowsy_counter = 0

            # Play alert if eyes closed for 20+ frames (~1 sec)
            if drowsy_counter > 15:
                threading.Thread(target=play_alert, daemon=True).start()
                cv2.putText(frame, "⚠️ DROWSINESS ALERT!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
