import cv2
import os
import numpy as np

def preprocess_img(img):
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.reshape(img, (1, 128, 128, 3))
    
    return img


def load_data(data_path):
    categories = os.listdir(data_path)


    X = []
    y = []
    for category in categories:
        path = os.path.join(data_path,category)
        label = categories.index(category)

        for img_name in os.listdir(path):
            img_path = os.path.join(path,img_name)
            img = cv2.imread(img_path)

            if img is not None:
                preprocess_img(img)
                X.append(img)
                y.append(label)
    X = np.array(X)
    y = np.array(y)
    return X,y             