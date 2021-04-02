import cv2,os
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from fuse_face_dataset.download_extract import download_url, url
import os



def load_face_data():
    data_path='7-2P-dataset'
    if not os.path.isdir(data_path):
        download_url(output_path='dataset.zip',url=url)

    categories=os.listdir(data_path)
    labels=[i for i in range(len(categories))]

    label_dict=dict(zip(categories,labels)) #empty dictionary

    img_size=100
    data=[]
    target=[]

    for category in categories:
        folder_path=os.path.join(data_path,category)
        img_names=os.listdir(folder_path)
            
        for img_name in img_names:
            img_path=os.path.join(folder_path,img_name)
            img=cv2.imread(img_path)

            try:
                gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)           
                #Converting the image into gray scale
                resized=cv2.resize(gray,(img_size,img_size))
                #resizing the gray scale into 50x50, since we need a fixed common size for all the images in the dataset
                data.append(resized)
                target.append(label_dict[category])
                #appending the image and the label(categorized) into the list (dataset)

            except Exception as e:
                print('Exception:',e)
                #if any exception raised, the exception will be printed here. And pass to the next image

    data=np.array(data)/1.0
    data=np.reshape(data,(data.shape[0],img_size,img_size,1))
    target=np.array(target)

    new_target=np_utils.to_categorical(target)


    train_data,test_data,train_target,test_target=train_test_split(data,new_target,test_size=0.2,stratify=new_target)
    
    return train_data,test_data,train_target,test_target

if __name__=='__main__':
    train_data,test_data,train_target,test_target = load_face_data()


