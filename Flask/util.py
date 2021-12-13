import base64
import json
import cv2
import numpy as np
from keras.models import load_model
import os

__class_name_to_number = {}
__class_number_to_name = {}

__model = None


def classify_image(image_base64_data, file_path=None):
    print('image_base64_data----', image_base64_data)
    # face_detector = detect_face_and_eyes(file_path, image_base64_data)
    # print('face detector content',face_detector)
    # if face_detector:

    # imgs = detect_face_and_eyes(file_path, image_base64_data)
    # imgs = np.array(imgs)
    #print('imgs----',imgs.shape)
    if file_path:
        img = cv2.imread(file_path)

    else:
        img = get_cv2_image_from_base64_string(image_base64_data)

    result = []
    # for img in imgs:
    #     scalled_raw_img = cv2.resize(img, (128, 128))
    #     #print('Scaled image----', scalled_raw_img)
    #     np_image = np.array(scalled_raw_img).astype('float32') / 255
    #     final = np.expand_dims(np_image, axis=0)
    #     #print('final----',final)
    scalled_raw_img = cv2.resize(img, (128, 128))
    #print('Scaled image----', scalled_raw_img)
    np_image = np.array(scalled_raw_img).astype('float32') / 255
    final = np.expand_dims(np_image, axis=0)
    #print('final----',final)

    result.append({
        'class': class_number_to_name(np.argmax(__model.predict(final)[0])),
        'class_probability': np.around(__model.predict(final) * 100, 2).tolist()[0],
        'class_dictionary': __class_name_to_number
    })
    return result

def class_number_to_name(class_num):
    # print('class num----',class_num)
    # print('__class_number_to_name[class_num]----',__class_number_to_name[class_num])
    return __class_number_to_name[class_num]


def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __class_name_to_number
    global __class_number_to_name

    with open("./class_dictionary.json", "r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v: k for k, v in __class_name_to_number.items()}

    global __model
    if __model is None:
        __model = load_model('Sports.h5')
    print("loading saved artifacts...done")


def get_cv2_image_from_base64_string(b64str):
    '''
    credit: https://stackoverflow.com/questions/33754935/read-a-base-64-encoded-image-from-memory-using-opencv-python-library
    :param uri:
    :return:
    '''
    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def detect_face_and_eyes(image_path, image_base64_data):
    face_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_frontalface_default.xml')
    # eye_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_eye.xml')
    # upper_body_cascade = cv2.CascadeClassifier(r'.\OpenCv\haarcascades\haarcascade_upperbody.xml')

    if image_path:
        img = cv2.imread(image_path)

    else:
        img = get_cv2_image_from_base64_string(image_base64_data)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # body = upper_body_cascade.detectMultiScale(
    #     gray,
    #     scaleFactor=1.01,
    #     minNeighbors=5,
    #     minSize=(30, 30),
    #     flags=cv2.CASCADE_SCALE_IMAGE
    # )


    cropped_faces = []
    for (x,y,w,h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            cropped_faces.append(roi_color)
            #print('cropped faces----',cropped_faces)
            # eyes = eye_cascade.detectMultiScale(roi_gray)
            #print('Eyes----',eyes)
            # if len(eyes) >= 2:
            #     result = True
    return cropped_faces


def get_b64_test_image_for_Sachin():
    with open("b64.txt") as f:
        return f.read()


if __name__ == '__main__':
    load_saved_artifacts()
    print(classify_image(None,r'C:\Users\ishan\data science\Projects\Sports Personality Classifier\Flask\test\m-2.jpg'))
    # for i in os.scandir(r'C:\Users\ishan\data science\Projects\Sports Personality Classifier\Flask\test'):
    #     print('path', i.path)
    #     result = classify_image(None, i.path)
    #     print('result class: ', result)


