# flask run --debug

from flask import Flask, render_template, request, Response,redirect, url_for
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
from main import try_image,send_message
import time
import secrets
import cv2
from ultralytics import YOLO

# Generate a random secret key
secret_key = secrets.token_hex(16)

app = Flask(__name__)
app.secret_key = secret_key

@app.route('/home')
def main():
    return render_template('home.html')

@app.route('/model-and-data')
def model_and_data():
    return render_template('model_data.html')

@app.route('/try-images', methods=['GET','POST'])
def try_images():
    if request.method == 'POST':

        # Get the uploaded image file
        uploaded_file = request.files['file']

        if uploaded_file.filename != '':

            # Save the image temporarily
            image_path = 'temp_image.jpg'
            uploaded_file.save(image_path)

            # Call the image detection function
            img_base64, result = try_image(image_path)
            return render_template('try_images.html', img_base64=img_base64, result=result)
    return render_template('try_images.html')



@app.route('/try-video', methods=['GET','POST'])
def try_video():
    form = TwilioForm()
    return render_template('try_video.html', form=form)


# Define the generate_frames() function outside of the video_feed() function
def generate_frames(model, cap, account_id, auth_token, to_number):
    while True:
        # Read frame from camera
        ret, frame = cap.read()

        if not ret:
            break

        # Perform model inference
        fire_detection = model(frame, conf=0.7)[0]

        # Draw predictions
        for detection in fire_detection.boxes.data.tolist():
            x1, y1, x2, y2, _, _ = detection
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(frame, 'Fire', (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Send message if fire is detected
        if fire_detection:
            send_message(account_id, auth_token, to_number)

        # Yield frame in HTTP response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/video_feed', methods=['GET', 'POST'])
def video_feed():
    form = TwilioForm()

    if form.validate_on_submit():
        account_id = form.account_id.data
        auth_token = form.auth_token.data
        to_number = form.to_number.data

        # Load model
        model = YOLO('best.pt')

        # Open camera
        cap = cv2.VideoCapture(0)

        # Call the generate_frames() function and return the response
        return Response(generate_frames(model, cap, account_id, auth_token, to_number), mimetype='multipart/x-mixed-replace; boundary=frame')

    return render_template('try_video.html', form=form)

@app.route('/feedback')
def feedback():
    return render_template('feedback.html')



# Webform for twilio
class TwilioForm(FlaskForm):
    account_id = StringField("Account ID", validators=[DataRequired()])
    auth_token = StringField("Auth token", validators=[DataRequired()])
    to_number = StringField("Enter your phone number (with country code please)", validators=[DataRequired()])
    submit = SubmitField("Submit")


if __name__ == '__main__':
    app.run(debug=True)
