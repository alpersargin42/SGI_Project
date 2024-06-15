import cv2
import numpy as np
from keras.models import load_model
from keras.utils import img_to_array

def analyze_faces():
    model = load_model("fer2013.h5")
    face_haar_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    cap = cv2.VideoCapture(0)

    emotion_colors = {
        'KIZGIN': (0, 0, 255),   # Kırmızı
        'NEFRET': (255, 255, 0),   # Sarı
        'KORKMUS': (255, 0, 0),   # Mavi
        'MUTLU': (0, 255, 0),   # Yeşil
        'DOGAL': (255, 165, 0),   # Turuncu
        'UZGUN': (128, 0, 128),   # Mor
        'SASKIN': (255, 192, 203)   # Pembe
    }

    while True:
        ret, test_img = cap.read()
        if not ret:
            continue

        gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

        for (x, y, w, h) in faces_detected:
            roi_gray = cv2.resize(gray_img[y:y + h, x:x + w], (96, 96))

            img_pixels = img_to_array(roi_gray) / 255.0
            img_pixels = np.expand_dims(img_pixels, axis=0)

            predictions = model.predict(img_pixels)
            max_index = np.argmax(predictions[0])
            predicted_emotion = ('KIZGIN', 'NEFRET', 'KORKMUS', 'MUTLU', 'DOGAL', 'UZGUN', 'SASKIN')[max_index]

            cv2.rectangle(test_img, (x, y), (x + w, y + h), emotion_colors[predicted_emotion], thickness=3)

            cv2.putText(test_img, predicted_emotion, (int(x), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, emotion_colors[predicted_emotion], 2)

        cv2.imshow('Analiz Edilen Ifade', cv2.resize(test_img, (640, 480)))

        if cv2.waitKey(10) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    analyze_faces()
