import cv2
import tensorflow as tf
import numpy as np

from tensorflow.keras.preprocessing.image import img_to_array

from project.ml.predict.utils import retrieve_model, retrieve_label_binarizers

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# CV2 requirements
faceCascade = cv2.CascadeClassifier('project/cv/haarcascade_frontalface_default.xml')
video_capture = cv2.VideoCapture(0)
# Note: check WebCam resolutions using ~> "uvcdynctrl -f"
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


def open_webcam(model_id):
    # Load model
    model = retrieve_model(model_id)

    # Load Label Binarizer
    age_lb, gender_lb = retrieve_label_binarizers()

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.FONT_HERSHEY_SIMPLEX
        )

        # Process each faces founded
        for (x, y, w, h) in faces:
            # Draw a rectangle around the faces
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Crop the detected face region
            face_crop = np.copy(frame[y:y+h, x:x+w])

            # Check that crop isn't "noise"
            if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
                continue

            # Pre-processing for gender detection model
            image = cv2.resize(face_crop, (96, 96))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_rgb = img_to_array(image)
            img_rgb = np.array(img_rgb, dtype="float") / 255.0
            img_rgb = np.expand_dims(img_rgb, axis=0)

            # Apply age/gender detection
            (age, gender) = model.predict(img_rgb)

            ageIdx = age[0].argmax()
            genderIdx = gender[0].argmax()

            age = age_lb.classes_[ageIdx]
            gender = gender_lb.classes_[genderIdx]

            # Write age label above face rectangle
            label = "{}".format(age)
            label_y = y - 10 if y - 10 > 10 else y + 10
            cv2.putText(frame, label, (x, label_y),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Write age label above face rectangle
            label = "{}".format('male' if gender == 1 else 'female')
            label_y = y - 30 if y - 30 > 30 else y + 30
            cv2.putText(frame, label, (x, label_y),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.namedWindow("my_window_name", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("my_window_name", 1500, 1500)
        cv2.imshow('my_window_name', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
