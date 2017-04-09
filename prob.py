import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression


TRAIN_DIR = ''
TEST_DIR = ''


IMG_SIZE = 50
LR = 1e - 3  #learning rate 0.001



MODEL_NAME = 'dogsvscats-{}-{}.model'.format(LR,'2conv-basic')


def label_img(img):
	word_label = img.split('.'[-3])  #dog.93.png

	if word_label = 'cat' : 
		return[1,0]
	elif word_label == 'dog' :
		return[0,1]


def create_train_data():
	training_data = []
	for img in tqdm(os.listdir(TRAIN_DIR)):
		label = label_img(img)
		path = os.path.join(TRAIN_DIR, img)

		img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE),(IMG_SIZE,IMG_SIZE))
		training_data.append([np.array(img), np.array(label)])


	shuffle(training_data)
	np.save('training_data.npy',training_data)
	return training_data


def process_test_data():
	testing_data = []
	for img in tqdm(os.listdir(TEST_DIR)):
		path = os.path.join(TEST_DIR,img)
		img_num = img.split('.')[0]
		img = cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(IMG_SIZE,IMG_SIZE))
		testing_data.append([np.array(img),img_num])


	np.save('test_data.npy',testing_data)
	return testing_data


train_data = create_train_data()

#if you already have train data:
#train_data = np.load('train_data.npy')


#convnet 

convnet = input_data(shape=[None,IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir = 'log')


