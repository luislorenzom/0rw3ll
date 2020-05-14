import cv2
import dlib
import numpy as np

from project.cv.recognizer import Recognizer, DEFAULT_FILE
from project.cv.centroid_tracker import CentroidTracker


# CV2 requirements
dnnFaceDetector = dlib.cnn_face_detection_model_v1('project/cv/mmod_human_face_detector.dat')

video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


def open_webcam():
    ct = CentroidTracker()
    recognizer = Recognizer(src=DEFAULT_FILE)
    # recognizer = Recognizer()
    try:
        while True:
            # Capture frame-by-frame
            ret, frame = video_capture.read()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = dnnFaceDetector(gray, 0)

            rects = []
            faces_crops = []
            # Process each faces founded
            for face in faces:
                x1 = face.rect.left()
                y1 = face.rect.top()
                x2 = face.rect.right()
                y2 = face.rect.bottom()

                # Add rectangle to compute the centroid
                rects.append((x1, y1, x2, y2))

                # Draw a rectangle around the faces
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Crop the detected face region
                faces_crops.append(np.copy(gray[y1:y2, x1:x2]))

            faces_crops.reverse()
            objects = ct.update(rects)

            if objects is not None:
                # Loop over the tracked objects
                for (objectID, centroid) in objects.items():
                    retrieved_id = None
                    train_required = False
                    if faces_crops:
                        face_crop = faces_crops.pop()
                        if recognizer.is_trained():
                            result = recognizer.predict(face_crop)
                            if result.get('confidence') < 50.0:
                                train_required = recognizer.update(objectID, face_crop)
                            else:
                                retrieved_id = result.get('face_id')
                        else:
                            train_required = recognizer.update(objectID, face_crop)
                        if train_required:
                            recognizer.train(objectID)
                    # Draw both the ID of the object and the centroid of the
                    # object on the output frame
                    text = "ID {}".format(retrieved_id if retrieved_id is not None else objectID)
                    cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
                # Display the resulting frame
                cv2.namedWindow("0rw3ll", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("0rw3ll", 750, 750)
                cv2.imshow('0rw3ll', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    except Exception as err:
        print(err)
    finally:
        # When everything is done, release the capture
        video_capture.release()
        cv2.destroyAllWindows()

        # Save hashing face model
        print('Persist model...')
        # recognizer.persist()
