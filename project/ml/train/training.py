import cv2
import datetime
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle

from time import time

from tensorflow.keras import Model, Input, layers
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

input_shape = (96, 96, 3)
filters = [32, 64, 128]
dropout_rates = [0.2, 0.4, 0.7]

# TRAINING PARAMS
EPOCHS = 50
LR = 1e-3
BS = 32
IMAGE_DIMS = (96, 96, 3)
DATA_PATH = 'project/ml/train/dataset/imdb/'

ts = lambda: str(int(time()))


def get_cnn():
    # Input
    input = inp = Input(shape=input_shape)
    # Three Conv2D+Max_Pool+Dropout layers
    for filter_num, dropout_rate in zip(filters, dropout_rates):
        inp = conv_block(input=inp, filter=filter_num, dropout=dropout_rate)
    # Dense Layer
    flat = layers.Flatten()(inp)
    dense = layers.Dense(units=1024, activation=tf.nn.relu)(flat)
    dropout = layers.Dropout(rate=0.4)(dense)
    # Age Dense
    age_dense = layers.Dense(units=90)(dropout)
    age_activation = layers.Activation(tf.nn.softmax, name="age_output")(age_dense)
    # Gender Dense and Activation
    gender_dense = layers.Dense(units=2)(dropout)
    gender_activation = layers.Activation(tf.nn.softmax, name="gender_output")(gender_dense)
    # Create model and Activation
    cnn_model = Model(inputs=input, outputs=[age_activation, gender_activation], name="0rw3ll")
    return cnn_model


def conv_block(input, filter, dropout):
    conv = layers.Conv2D(filters=filter, kernel_size=[5, 5], padding="same", activation="relu")(input)
    max_pool = layers.MaxPooling2D(pool_size=[2, 2], strides=2)(conv)
    dropout = layers.Dropout(rate=dropout)(max_pool)
    return dropout


def transform_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = img_to_array(image)
    return image


def training():
    # Load dataset
    dataset = pd.read_csv(DATA_PATH + 'imdb.csv')

    # Transform images
    dataset['filename'] = DATA_PATH + 'images/' + dataset['filename']

    # TODO: remove (this is because i just have 32 gb of RAM)
    dataset = dataset[:9000]

    # Train/Test split
    # ----------------------------------------------------------------------
    # https://www.kernel.org/doc/Documentation/vm/overcommit-accounting
    # echo 1 > /proc/sys/vm/overcommit_memory
    # ----------------------------------------------------------------------
    # Image np style
    data = np.array(list(dataset['filename'].apply(transform_image)), dtype="float") / 255.0

    # Age
    age_lb = LabelBinarizer()
    age_labels = age_lb.fit_transform(dataset['age'])

    # Gender
    gender_lb = LabelBinarizer()
    gender_labels = gender_lb.fit_transform(dataset['gender'])

    split = train_test_split(data, age_labels, gender_labels, test_size=0.2, random_state=42)
    (trainX, testX, trainCategoryY, testCategoryY, trainColorY, testColorY) = split

    # Training
    losses = {"age_output": "categorical_crossentropy", "gender_output": "sparse_categorical_crossentropy"}
    loss_weights = {"age_output": 1.0, "gender_output": 1.0}
    opt = Adam(lr=LR, decay=LR / EPOCHS)
    model = get_cnn()
    model.compile(optimizer=opt, loss=losses, loss_weights=loss_weights, metrics=['accuracy'])

    # Add TensorBoard
    log_dir = "artifacts/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    h = model.fit(trainX,
                  {"age_output": trainCategoryY, "gender_output": trainColorY},
                  validation_data=(testX, {"age_output": testCategoryY, "gender_output": testColorY}),
                  epochs=EPOCHS,
                  verbose=1,
                  callbacks=[tensorboard_callback])

    # Persist model
    model.save('artifacts/model/{}'.format(ts()))

    # Persist Age Label Binarizer
    f = open('artifacts/label_binarizer/age_lb.bin', "wb")
    f.write(pickle.dumps(age_lb))
    f.close()

    # Persist Gender Label Binarizer
    f = open('artifacts/label_binarizer/gender_lb.bin', "wb")
    f.write(pickle.dumps(gender_lb))
    f.close()
