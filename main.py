import argparse

from project.cv.face_detector import open_webcam
from project.ml.train.training import training

parser = argparse.ArgumentParser(description='0rw3ll')

parser.add_argument('-t', '--train', action="store_true", default=False, help='Train cnn model')
parser.add_argument('-d', '--deploy', action="store", dest="model_id", type=str, help='Deploy model on WebCam')

args = parser.parse_args()

if args.train:
    training()
elif args.model_id is not None:
    open_webcam(args.model_id)
else:
    parser.error('Orw3ll requires train mode (-t) or deploy mode (-d)')
