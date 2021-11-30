from flask import Flask, Response, request, url_for, flash
from flask.templating import render_template
from werkzeug.utils import redirect
import cv2
import numpy as np
from dotenv import load_dotenv
import base64
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import pickle

from config import Config
from core.face_recognizer import FaceRecognizer
from core.face_encoder import FaceEncoder
from core.face_detector import FaceDetector

load_dotenv()

db = SQLAlchemy()

app = Flask(__name__)
app.config.from_object(Config)

db.init_app(app)
migrate = Migrate(app, db)

from api.models import User

face_detector = FaceDetector()
face_encoder = FaceEncoder()
face_recognizer = FaceRecognizer()

camera = cv2.VideoCapture(0)

print("All class sucessfully loaded!")


def get_frame():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            result = ""
            try:
                extracted_faces = face_detector.extract_face(image_array=frame)
                for extracted_face in extracted_faces:
                    face_embedding = face_encoder.get_embedding(image_array=extracted_face)
                    print(face_embedding)
                    checked_in_user = get_check_in_user(face_embedding)
                    if checked_in_user is None:
                        result = "No user found, please add the user to database"
                    else:
                        print(checked_in_user.name)
                        result = result + "," + checked_in_user.name
            except:
                result = "Face not found. Please try again!"
            ret, buffer = cv2.imencode('.jpg', img=frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


def get_check_in_user(face_embedding):
    with app.app_context():
        users = User().query.all()

    similarities = [face_recognizer.compare(
        face_embedding, pickle.loads(user.encoding)) for user in users]

    print([(users[idx].name, similarity) for idx, similarity in enumerate(similarities)])

    max_similarity = max(similarities)

    print("Maximum similarities: {}".format(max_similarity))

    max_similarity_index = similarities.index(max_similarity)

    print(type(max_similarity > 0.99))

    if max_similarity > 0.97:
        checked_in_user = users[max_similarity_index]
        print("User: {}".format(checked_in_user))
        return checked_in_user

    return None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)