import cv2
from ultralytics import YOLO
from twilio.rest import Client
import base64

def try_image(img_path='fire.jpg'):
    # Load Model
    model = YOLO('best.pt')

    # Read Image
    image = cv2.imread(img_path)

    # Resize images to be all at the same size
    image = cv2.resize(image, (800,600))

    # Perform Detection
    fire_detection = model(image, conf=0.5)[0]
    detections = []

    for detection in fire_detection.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        detections.append([x1, y1, x2, y2, score])

    # Draw Predictions
    for fire_box in detections:
        x1, y1, x2, y2, _ = fire_box
        # Bounding Box
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        # Put Text
        cv2.putText(image, 'Fire', (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Convert image to base64
        _, img_encoded = cv2.imencode('.png', image)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')

        return img_base64, detections

def send_message(account_id, auth_token, to_number):
    client = Client(account_id, auth_token)
    message = client.messages.create(
        from_= 'whatsapp:+14155238886',
        body = 'Fire detected! Your attention is required.',
        to = 'whatsapp:' + to_number
    )
    print("Message sent successfully")