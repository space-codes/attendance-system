import os
import pickle
import numpy as np
import cv2
import mtcnn
from keras.models import load_model
from utils.utils import get_face, get_encode, l2_normalizer, normalize

# hyper-parameters
encoder_model = 'facenet_keras.h5'
people_dir = 'images'
encodings_path = 'encodings.pkl'
required_size = (160, 160)

face_detector = mtcnn.MTCNN()
face_encoder = load_model(encoder_model)

encoding_dict = dict()

encode= []

for img in people_dir:
    img = cv2.imread(img)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detector.detect_faces(img_rgb)
    if results:
        res = max(results, key=lambda b: b['box'][2] * b['box'][3])
        face, _, _ = get_face(img_rgb, res['box'])

        face = normalize(face)
        face = cv2.resize(face, required_size)
        encoded = face_encoder.predict(np.expand_dims(face, axis=0))[0]
        encode.append(encoded)

    if encode:
        encode = np.sum(encode, axis=0)
        encode = l2_normalizer.transform(np.expand_dims(encode, axis=0))[0]
        person_name = os.path.splitext(img)[0]
        encoding_dict[person_name] = encode

for key in encoding_dict.keys():
    print(key)

with open(encodings_path, 'bw') as file:
    pickle.dump(encoding_dict, file)