#!/usr/bin/env python
# coding: utf-8

"""
for subset in `seq 0 9`
do
python -W ignore infinite_generator_3D.py \
--fold $subset \
--scale 32 \
--data /mnt/dataset/shared/zongwei/LUNA16 \
--save generated_cubes
done
"""

# In[1]:


import warnings
warnings.filterwarnings('ignore')
import os
'''import keras
print("Keras = {}".format(keras.__version__))
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}'''

import sys
import math
import random

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

from tqdm import tqdm
from sklearn import metrics
from optparse import OptionParser
from glob import glob
from skimage.transform import resize

from DynamicMRI import DynamicMRI

sys.setrecursionlimit(40000)

parser = OptionParser()

#parser.add_option("--fold", dest="fold", help="fold of subset", default=None, type="int")
parser.add_option("--input_rows", dest="input_rows", help="input rows", default=128, type="int")
parser.add_option("--input_cols", dest="input_cols", help="input cols", default=128, type="int")
parser.add_option("--input_deps", dest="input_deps", help="input deps", default=32, type="int")
parser.add_option("--crop_rows", dest="crop_rows", help="crop rows", default=128, type="int")
parser.add_option("--crop_cols", dest="crop_cols", help="crop cols", default=128, type="int")
parser.add_option("--crop_deps", dest="crop_deps", help="crop deps", default=32, type="int")
parser.add_option("--data", dest="data", help="the directory of LUNA16 dataset", default=None, type="string")
parser.add_option("--save", dest="save", help="the directory of processed 3D cubes", default=None, type="string")
parser.add_option("--scale", dest="scale", help="scale of the generator", default=32, type="int")
(options, args) = parser.parse_args()
#fold = options.fold

seed = 1
random.seed(seed)

assert options.data is not None
assert options.save is not None
#assert options.fold >= 0 and options.fold <= 9

if not os.path.exists(options.save):
    os.makedirs(options.save)

class setup_config():
    #hu_max = 1000.0
    #hu_min = -1000.0
    #HU_thred = (-150.0 - hu_min) / (hu_max - hu_min)
    def __init__(self, 
                 input_rows=None, 
                 input_cols=None,
                 input_deps=None,
                 crop_rows=None, 
                 crop_cols=None,
                 crop_deps=None,
                 len_border=None,
                 len_border_z=None,
                 scale=None,
                 DATA_DIR=None,
                 train_fold=[1,2,3,4,5,6,7,8,9,10,11,12,14,15,16,17,18,20,21,22,23,24,25,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55],
                 valid_fold=[56,57,58,59,61,62,63,64,66,68,69],
                 test_fold=[],
                 #len_depth=None,
                 #lung_min=0.7,
                 #lung_max=1.0,
                ):
        self.input_rows = input_rows
        self.input_cols = input_cols
        self.input_deps = input_deps
        self.crop_rows = crop_rows
        self.crop_cols = crop_cols
        self.crop_deps = crop_deps
        self.len_border = len_border
        self.len_border_z = len_border_z
        self.scale = scale
        self.DATA_DIR = DATA_DIR
        self.train_fold = train_fold
        self.valid_fold = valid_fold
        self.test_fold = test_fold
        #self.len_depth = len_depth
        #self.lung_min = lung_min
        #self.lung_max = lung_max

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")



config = setup_config(input_rows=options.input_rows,
                      input_cols=options.input_cols,
                      input_deps=options.input_deps,
                      crop_rows=options.crop_rows,
                      crop_cols=options.crop_cols,
                      crop_deps=options.crop_deps,
                      scale=options.scale,
                      len_border=10,
                      len_border_z=0,
                      #len_depth=3,
                      #lung_min=0.7,
                      #lung_max=0.15,
                      DATA_DIR=options.data,
                     )
config.display()

def infinite_generator_from_one_volume(config, img_array):
    #size_x, size_y, size_z = img_array.shape
    size_z, size_y, size_x = img_array.shape
    if size_z-config.crop_deps-1-config.len_border_z < config.len_border_z:
        return None
    
    #img_array[img_array < config.hu_min] = config.hu_min
    #img_array[img_array > config.hu_max] = config.hu_max
    #img_array = 1.0*(img_array-config.hu_min) / (config.hu_max-config.hu_min)
    
    slice_set = np.zeros((config.scale, config.input_deps, config.input_cols, config.input_rows), dtype=float)
    
    num_pair = 0
    cnt = 0
    while True:
        cnt += 1
        if cnt > 50 * config.scale and num_pair == 0:
            return None
        elif cnt > 50 * config.scale and num_pair > 0:
            return np.array(slice_set[:num_pair])

        start_x = random.randint(0+config.len_border, size_x-config.crop_rows-1-config.len_border)
        start_y = random.randint(0+config.len_border, size_y-config.crop_cols-1-config.len_border)
        start_z = random.randint(0+config.len_border_z, size_z-config.crop_deps-1-config.len_border_z)
        
        crop_window = img_array[start_z : start_z+config.crop_deps,
                                start_y : start_y+config.crop_cols,
                                start_x : start_x+config.crop_rows
                               ]
        if config.crop_rows != config.input_rows or config.crop_cols != config.input_cols:
            crop_window = resize(crop_window, 
                                 (config.crop_deps, config.input_cols, config.input_rows), 
                                 preserve_range=True,
                                )
        
        #t_img = np.zeros((config.input_rows, config.input_cols, config.input_deps), dtype=float)
        #d_img = np.zeros((config.input_rows, config.input_cols, config.input_deps), dtype=float)
        
        '''for d in range(config.input_deps):
            for i in range(config.input_rows):
                for j in range(config.input_cols):
                    for k in range(config.len_depth):
                        if crop_window[i, j, d+k] >= config.HU_thred:
                            t_img[i, j, d] = crop_window[i, j, d+k]
                            d_img[i, j, d] = k
                            break
                        if k == config.len_depth-1:
                            d_img[i, j, d] = k
                            
        d_img = d_img.astype('float32')
        d_img /= (config.len_depth - 1)
        d_img = 1.0 - d_img
        
        if np.sum(d_img) > config.lung_max * config.input_rows * config.input_cols * config.input_deps:
            continue'''
        
        slice_set[num_pair] = crop_window
        
        num_pair += 1
        if num_pair == config.scale:
            break
            
    return np.array(slice_set)


def get_self_learning_data(fold, config):
    slice_set = []
    #for index_subset in fold:
    subset_path = os.path.join(config.DATA_DIR, '%02d' % (fold))
    #[os.path.join(subset_path, f) for f in os.listdir(subset_path) 
    #            if os.path.isdir(os.path.join(subset_path,f))]
    dcemr = DynamicMRI(subset_path)
    for phase in dcemr.phases:
        #itk_img = phase.image
        img_array = phase.numpy()
        #img_array = ((img_array.astype(np.float32)-img_array.min()) * 2.0 / (img_array.max()-img_array.min()))-1.0
        img_array = (img_array.astype(np.float32)-img_array.min()) / (img_array.max()-img_array.min())
        #print(img_array.shape)
        #img_array = img_array.transpose(2, 1, 0)
        
        x = infinite_generator_from_one_volume(config, img_array)
        if x is not None:
            slice_set.extend(x)
            
    return np.array(slice_set)
    #return 0


#print(">> Fold {}".format(fold))
for fold in config.train_fold:
#for fold in config.valid_fold:
    print(">> Fold {}".format(fold))
    cube = get_self_learning_data(fold, config)
    if cube.size != 0:
        print("cube: {} | {:.2f} ~ {:.2f}".format(cube.shape, np.min(cube), np.max(cube)))
        np.save(os.path.join(config.DATA_DIR,
                            options.save, 
                            "bat_"+str(config.scale)+
                            "_"+str(config.input_rows)+
                            "x"+str(config.input_cols)+
                            "x"+str(config.input_deps)+
                            "_"+str(fold)+".npy"), 
                cube,
            )