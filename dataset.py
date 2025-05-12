import os
import os.path
import numpy as np
import random
import h5py
import torch
import cv2
import glob
import torch.utils.data as udata

def normalize(data):    ###定义数据预处理函数，将数据归一化到0-1范围
    return data/255.

def prepare_data(tr_path, vl_path,noiseL=15):
    # train
    print('process training data')
    files = glob.glob(os.path.join(tr_path, '*.png'))
    files.sort()
    # create a new HDF5 file, 创建新文件写，已经存在的文件会被覆盖掉
    file_tr = f'mytrain{noiseL}.h5'
    h5f = h5py.File(file_tr, 'w')
    train_num = 0
    for i in range(len(files)):
        print("file: %s" % files[i])
        img = cv2.imread(files[i])
        img = np.expand_dims(img[:, :, 0], 0)
        img = np.float32(normalize(img))

        if img.shape[1] < img.shape[2]:
            # 如果后两位的第一个元素小于第二个，则交换维度
            img = np.transpose(img, (0, 2, 1))  # 交换行和列

        h5f.create_dataset(str(train_num), data=img)
        train_num += 1
    h5f.close()
    # val
    print('\nprocess validation data')
    files.clear()
    files = glob.glob(os.path.join(vl_path, '*.png'))
    files.sort()
    file_vl = f'myval{noiseL}.h5'
    h5f = h5py.File(file_vl, 'w')
    val_num = 0
    for i in range(len(files)):
        print("file: %s" % files[i])
        img = cv2.imread(files[i])
        img = np.expand_dims(cv2.resize(img[:, :, 0], (256, 256)), 0)
        img = np.float32(normalize(img))
        h5f.create_dataset(str(val_num), data=img)
        val_num += 1
    h5f.close()
    print('training set, # samples %d\n' % train_num)
    print('val set, # samples %d\n' % val_num)


class Dataset(udata.Dataset):
    def __init__(self, train=True,noiseL=15):
        super(Dataset, self).__init__()
        self.train = train
        self.file_tr = f'mytrain{noiseL}.h5'
        self.file_vl = f'myval{noiseL}.h5'

        if self.train:
            h5f = h5py.File(self.file_tr, 'r')
        else:
            h5f = h5py.File(self.file_vl, 'r')
        self.keys = list(h5f.keys())
        random.shuffle(self.keys)
        h5f.close()
    def __len__(self):
        return len(self.keys)
    def __getitem__(self, index):
        if self.train:
            h5f = h5py.File(self.file_tr, 'r')
        else:
            h5f = h5py.File(self.file_vl, 'r')
        key = self.keys[index]
        data = np.array(h5f[key])
        h5f.close()
        return torch.Tensor(data)

if __name__ == "__main__":

    data_path = './data/'
    patch_size = 40
    stride = 10
    prepare_data(data_path, patch_size, stride, aug_times=1)