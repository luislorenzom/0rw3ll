import cv2
import numpy as np

DEFAULT_FILE = 'artifacts/trainer/trainer.yml'


class Face:
    def __init__(self, train_threshold=60):
        self.frames = []
        self.train_threshold = train_threshold


class Recognizer:
    def __init__(self, src=None):
        self.faces = dict()
        self.trained = False
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        if src is not None:
            self.face_recognizer.read(src)
            self.trained = True

    def update(self, _id, face):
        if _id not in self.faces:
            self.faces[_id] = Face()
        self.faces.get(_id).frames.append(face)
        self.faces.get(_id).train_threshold -= 1
        return self.faces.get(_id).train_threshold <= 0

    def train(self, _id):
        labels = np.array([_id]*60)
        self.face_recognizer.train(self.faces.get(_id).frames, labels)
        del self.faces[_id]
        self.trained = True

    def predict(self, face):
        face_id, confidence = self.face_recognizer.predict(face)
        return {
            'face_id': face_id,
            'confidence': confidence
        }

    def persist(self, src=DEFAULT_FILE):
        with open(src, 'w') as _:
            self.face_recognizer.write(src)

    def is_trained(self):
        return self.trained
