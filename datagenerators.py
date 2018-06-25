# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 11:40:20 2018

@author: Abhijit
"""

import numpy as np
import cv2
import glob 
import os
from keras.preprocessing.image import ImageDataGenerator


class doublet_data(ImageDataGenerator):
    """A basic keras generator for generating pair of images for siamese networks
    input_dir = path to top level directory containing data folders (similar to flow_from_directory)
    """
    def __init__(self,
                 input_dir,
                 rotation_range=0.,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 shear_range=0.,
                 zoom_range=0.,
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None,
                 preprocessing_function=None,
                 batch_size=32,
                 input_shape=(),
                 flatten = False,
                 num_channels = 3,
                 seed = 5000,
                 dtype=np.float32):
        
        np.random.seed(seed)
        assert len(input_shape) == 0 or len(input_shape) == 2, "The input shape must be a tuple of length 2"
        self.flatten = flatten
        self.input_dir = input_dir
        self.input_shape=input_shape
        self.num_channels = num_channels
        assert num_channels == 3 or num_channels == 1,"The number of channels must be either 1 or 3"
        assert batch_size > 1, "batch size must be greater than 1"
        self.sub_dirs = [os.path.join(input_dir,x) for x in os.listdir(input_dir)]
        self.batch_size = batch_size
        self.dtype=dtype
        for sub_dir in self.sub_dirs:
            print("number of images in {} is {}".format(os.path.basename(sub_dir),len(os.listdir(sub_dir))))
        
        super().__init__(
                       rotation_range=rotation_range,
                       width_shift_range=width_shift_range,
                       height_shift_range=height_shift_range,
                       shear_range=shear_range,
                       zoom_range=zoom_range,
                       horizontal_flip=horizontal_flip,
                       vertical_flip=vertical_flip,
                       rescale=rescale,
                       preprocessing_function=preprocessing_function
                       ) 
        
    def generate_data(self):
        while True:
            all_images_path = []
            for sub_dir in self.sub_dirs:
                all_images_path.append(glob.glob(os.path.join(sub_dir,'*')))
            all_images_path = np.array(all_images_path)
            X1 = []
            X2 = []
            Y = []
            available_choices =  [i for i in range(len(all_images_path))]
            for i in range(self.batch_size):
                choice_index = np.random.choice(available_choices)
                if i%2 == 0: # take genuine pair
                    img1 = cv2.imread(np.random.choice(all_images_path[choice_index]))
                    img2 = cv2.imread(np.random.choice(all_images_path[choice_index]))
                    Y.append(1)
                else:
                    other_choices = list(set(available_choices)-set([choice_index]))
                    other_choice_index = np.random.choice(other_choices)
                    img1 = cv2.imread(np.random.choice(all_images_path[choice_index]))
                    img2 = cv2.imread(np.random.choice(all_images_path[other_choice_index]))
                    Y.append(0)
                img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
                img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
                
                # convert to float before preprocessing
                img1 = img1.astype(self.dtype)
                img2 = img2.astype(self.dtype)
                
                if self.preprocessing_function:
                    img1 = self.preprocessing_function(img1)
                    img2 = self.preprocessing_function(img2)
                    
                # apply image augmentations if any
                img1 = self.random_transform(img1)
                img2 = self.random_transform(img2)
                
                if len(self.input_shape) == 2:
                    img1 = cv2.resize(img1,self.input_shape)
                    img2 = cv2.resize(img2,self.input_shape)
                
                if self.num_channels == 1:
                    img1 = img1[:,:,0].reshape(img1.shape[0],img1.shape[1],1)
                    img2 = img2[:,:,0].reshape(img2.shape[0],img2.shape[1],1)
                if self.flatten:
                    img1 = np.reshape(img1,-1)
                    img2 = np.reshape(img2,-1)

                X1.append(img1)
                X2.append(img2)

            yield ([np.array(X1),np.array(X2)],np.array(Y))
            
###############################################################################
            
class triplet_data(ImageDataGenerator):
    """A basic keras triplet generator
    NOTE: The generation is random, It does not implement hard triplets as given in the Facenet paper
    input_dir = path to top level directory containing data folders (similar to flow_from_directory)
    The returned Y is all 0 because y_true is not used in triplet loss
    """
    def __init__(self,
                 input_dir,
                 rotation_range=0.,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 shear_range=0.,
                 zoom_range=0.,
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None,
                 preprocessing_function=None,
                 batch_size=32,
                 input_shape=(),
                 flatten = False,
                 num_channels = 3,
                 seed = 5000,
                 dtype=np.float32):
        np.random.seed(seed)
        assert len(input_shape) == 0 or len(input_shape) == 2, "The input shape must be a tuple of length 2"
        self.flatten = flatten
        self.input_dir = input_dir
        self.input_shape=input_shape
        self.num_channels = num_channels
        assert num_channels == 3 or num_channels == 1,"The number of channels must be either 1 or 3"
        assert batch_size > 1, "batch size must be greater than 1"
        self.sub_dirs = [os.path.join(input_dir,x) for x in os.listdir(input_dir)]
        self.batch_size = batch_size
        self.dtype = dtype
        for sub_dir in self.sub_dirs:
            print("number of images in {} is {}".format(os.path.basename(sub_dir),len(os.listdir(sub_dir))))
        
        super().__init__(
                       rotation_range=rotation_range,
                       width_shift_range=width_shift_range,
                       height_shift_range=height_shift_range,
                       shear_range=shear_range,
                       zoom_range=zoom_range,
                       horizontal_flip=horizontal_flip,
                       vertical_flip=vertical_flip,
                       rescale=rescale,
                       preprocessing_function=preprocessing_function
                       ) 
        
    def generate_data(self):
        while True:
            all_images_path = []
            for sub_dir in self.sub_dirs:
                all_images_path.append(glob.glob(os.path.join(sub_dir,'*')))
            all_images_path = np.array(all_images_path)
            anchor_list = []
            positive_list = []
            negative_list = []
            Y = []

            available_choices =  [i for i in range(len(all_images_path))]
            for i in range(self.batch_size):
                choice_index = np.random.choice(available_choices)
                anchor = cv2.imread(np.random.choice(all_images_path[choice_index]))
                positive = cv2.imread(np.random.choice(all_images_path[choice_index]))

                other_choices = list(set(available_choices)-set([choice_index]))
                other_choice_index = np.random.choice(other_choices)

                negative = cv2.imread(np.random.choice(all_images_path[other_choice_index]))
                Y.append(0) #Y is not important here so append 0
                anchor = cv2.cvtColor(anchor,cv2.COLOR_BGR2RGB)
                positive = cv2.cvtColor(positive,cv2.COLOR_BGR2RGB)
                negative = cv2.cvtColor(negative,cv2.COLOR_BGR2RGB)
                
                # convert to float before preprocessing
                anchor = anchor.astype(self.dtype)
                positive = positive.astype(self.dtype)
                negative = negative.astype(self.dtype)
                
                if self.preprocessing_function:
                    anchor = self.preprocessing_function(anchor)
                    positive = self.preprocessing_function(positive)
                    negative = self.preprocessing_function(negative)
                    
                # apply image augmentations if any
                anchor = self.random_transform(anchor)
                positive = self.random_transform(positive)
                negative = self.random_transform(negative)
                
                if len(self.input_shape) == 2:
                    anchor = cv2.resize(anchor,self.input_shape)
                    positive = cv2.resize(positive,self.input_shape)
                    negative = cv2.resize(negative,self.input_shape)
                    
                if self.num_channels == 1:
                    anchor = anchor[:,:,0].reshape(anchor.shape[0],anchor.shape[1],1)
                    positive = positive[:,:,0].reshape(positive.shape[0],positive.shape[1],1)
                    negative = negative[:,:,0].reshape(negative.shape[0],negative.shape[1],1)
                    
                if self.flatten:
                    anchor = np.reshape(anchor,-1)
                    positive = np.reshape(positive,-1)
                    negative = np.reshape(negative,-1)

                anchor_list.append(anchor)
                positive_list.append(positive)
                negative_list.append(negative)
            yield ([np.array(anchor_list),np.array(positive_list),np.array(negative_list)],np.array(Y))