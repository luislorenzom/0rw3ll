import pickle
import tensorflow as tf


def retrieve_model(model_id):
    return tf.keras.models.load_model("artifacts/model/{}".format(model_id))


def retrieve_label_binarizers():
    age_lb = pickle.loads(open('artifacts/label_binarizer/age_lb.bin', "rb").read())
    gender_lb = pickle.loads(open('artifacts/label_binarizer/gender_lb.bin', "rb").read())
    return age_lb, gender_lb

# TODO
def predict(model, img, age_lb):
    (age, gender) = model.predict(img)

    ageIdx = age[0].argmax()
    genderIdx = gender[0].argmax()

    #age = age_lb.classes_[ageIdx]
    #gender = gender_lb.classes_[genderIdx]
