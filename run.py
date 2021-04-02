from fuse_face_dataset import data_loader

if __name__=='__main__':
    train_data,test_data,train_target,test_target = data_loader.load_face_data()