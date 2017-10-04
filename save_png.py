'''
    tranfer the test dataset output to imgs
'''
# coding:UTF-8
from keras.preprocessing.image import array_to_img
import numpy as np
import glob
import os, os.path

def save_output(default_dir):
    '''
    save test dataset into imgs into a default dir
    '''

    imgs_train = np.load(file= "./imgs_mask_test.npy")
    imgs_train = imgs_train.astype('float32')

    pictures = np.shape(imgs_train)[0]
    imgs = glob.glob("train/*.tif")
    print pictures
    for i in range(pictures):
        img_tmp = array_to_img(imgs_train[i])

        nr = imgs[i].split("/")[1].split(".")[0]
        print nr
        img_tmp.save(default_dir+"{}.".format(nr)+"png")

if __name__ == "__main__":
    default_dir = "./test_result/"
    save_output(default_dir)