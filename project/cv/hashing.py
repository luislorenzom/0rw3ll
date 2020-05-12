import cv2
import dlib
import numpy as np

# CV2 requirements
dnnFaceDetector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')
video_capture = cv2.VideoCapture(0)

video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


def open_webcam():
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = dnnFaceDetector(gray, 0)

        # Process each faces founded
        for face in faces:
            x1 = face.rect.left()
            y1 = face.rect.top()
            x2 = face.rect.right()
            y2 = face.rect.bottom()

            # Draw a rectangle around the faces
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Display the resulting frame
        cv2.namedWindow("0rw3ll", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("0rw3ll", 750, 750)
        cv2.imshow('0rw3ll', frame)

        # Crop the detected face region
        face_crop = np.copy(frame[y1:y2, x1:x2])
        # TODO save facecrop to train the recogniser

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    open_webcam()