import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)

def draw_landmarks(frame, x, y, w, h):

    landmarks = {
        "left eye": (x + w // 4, y + h // 3),
        "right eye": (x + 3 * w // 4, y + h // 3),
        "nose": (x + w // 2, y + h // 2),
        "left cheek": (x + w // 5, y + 2 * h // 3),
        "right cheek": (x + 2 * w // 3, y + 2 * h // 3),
        "head": (x + w // 2, y + h // 0),
        "chin": (x + w // 2, y + h // 1),
        "lips": (x + w // 2, y + h*2 //3)
    }

    for name, point in landmarks.items():
        cv2.circle(frame, point, 5, (168, 134, 155), -1)
        cv2.putText(frame, name, (point[0] + 5, point[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (168, 134, 155), 2)
        draw_landmarks(frame, x, y, w, h)

    cv2.imshow("Simple Facial Landmark Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()