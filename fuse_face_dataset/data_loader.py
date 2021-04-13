import cv2,os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.utils import np_utils
from fuse_face_dataset.download_extract import download_url, url
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img


def load_face_data():
    data_path='7-2P-dataset_renamed'
    if not os.path.isdir(data_path):
        download_url(output_path='dataset.zip',url=url)

    categories=os.listdir(data_path)
    labels=[i for i in range(len(categories))]
    img_size=224
    data=[]
    target=[]

    for category in categories:
        folder_path=os.path.join(data_path,category)
        img_names=os.listdir(folder_path)
            
        for img_name in img_names:
            img_path=os.path.join(folder_path,img_name)
            image = load_img(img_path, target_size=(img_size, img_size))
            image = img_to_array(image)
            image = preprocess_input(image)
            
            data.append(image)
            target.append(category)

    lb = LabelBinarizer()
    labels = lb.fit_transform(target)
    labels = to_categorical(labels)

    data = np.array(data, dtype="float32")
    labels = np.array(labels)

    trainX, testX, trainY, testY = train_test_split(data, labels,test_size=0.20, stratify=labels, random_state=42)
    return trainX, testX, trainY, testY

if __name__=='__main__':
    train_data,test_data,train_target,test_target = load_face_data()


