from matplotlib import image, pyplot
from glob import glob
from numpy import asarray
from skimage.transform import resize as imresize
from PIL import Image
import pickle
import os
import urllib
import gzip
import numpy as np
import scipy
import cv2

categories = ['aeroplane', 'bicycle', 'bus', 'car', 'horse', 'knife', 'motorcycle', 'person', 'plant', 'skateboard', 'train', 'truck', 'other']

trainX = []
trainY = []
testX = []
testY = []


trainRoot = "C:/Users/Shir/Desktop/School/CZ4041_Machine_Learning/Actual_dataset/train/"
testRoot = "C:/Users/Shir/Desktop/School/CZ4041_Machine_Learning/Actual_dataset/validation/"

for i in range (len(categories)):
    trainX = []
    trainY = []
    testX = []
    testY = []
    print("Setting training " + categories[i])
    trainPath = trainRoot + categories[i] + "/*.png"
    testPath = testRoot + categories[i] + "/*.jpg"
    for j, file in enumerate(glob(trainPath)):
        image = cv2.imread(file)
        if image.ndim == 1:
            image = cv2.merge((image, image, image))
        res_image = cv2.resize(image, dsize=(100,100), interpolation=cv2.INTER_CUBIC)
        trainX.append(res_image)
        trainY.append(i)
    print("Setting testing " + categories[i])
    for j, file in enumerate(glob(testPath)):
        image = cv2.imread(file)
        if image.ndim == 1:
            image = cv2.merge((image, image, image))
        res_image = cv2.resize(image, dsize=(100,100), interpolation=cv2.INTER_CUBIC)
        testX.append(res_image)
        testY.append(i)
    #trainX = np.array(trainX, dtype = 'float32')
    #trainY = np.array(trainY, dtype = 'float32')
    with open('C:/Users/Shir\Desktop/School/CZ4041_Machine_Learning/Actual_dataset//Pickle/train_{}.npz'.format(categories[i]),'wb') as f:
        print("Saving training " + categories[i])
        np.savez_compressed(f, x = trainX, y = trainY)

    with open('C:/Users/Shir\Desktop/School/CZ4041_Machine_Learning/Actual_dataset//Pickle/test_{}.npz'.format(categories[i]),'wb') as f:
        print("Saving testing " + categories[i])
        np.savez_compressed(f, x = testX, y = testY)    










##data = DataLoader(img_res=(32, 32))
##
##imgs_A, labels_A = data.load_data(domain="A", batch_size=128)
##imgs_B, labels_B = data.load_data(domain="B", batch_size=128)
