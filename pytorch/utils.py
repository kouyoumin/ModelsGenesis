from __future__ import print_function
import math
import os
import random
import copy
import scipy
import imageio
import string
import numpy as np
import cv2
from skimage.transform import resize
try:  # SciPy >= 0.19
    from scipy.special import comb
except ImportError:
    from scipy.misc import comb

def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """

    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i

def bezier_curve(points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.

       Control points should be a list of lists, or list of tuples
       such as [ [1,1], 
                 [2,3], 
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000

        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([ bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)   ])
    
    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals

def data_augmentation(x, y, prob=0.5, random_state=None):
    # augmentation by flipping
    cnt = 3
    if isinstance(random_state, np.random.RandomState):
         while random_state.random() < prob and cnt > 0:
            degree = random_state.choice([0, 1, 2])
            x = np.flip(x, axis=degree)
            y = np.flip(y, axis=degree)
            cnt = cnt - 1
    else:
       while random.random() < prob and cnt > 0:
            degree = random.choice([0, 1, 2])
            x = np.flip(x, axis=degree)
            y = np.flip(y, axis=degree)
            cnt = cnt - 1

    return x, y

def nonlinear_transformation(x, prob=0.5, random_state=None):
    if isinstance(random_state, np.random.RandomState):
        if random_state.random() >= prob:
            return x
        points = [[0, 0], [random_state.random(), random_state.random()], [random_state.random(), random_state.random()], [1, 1]]
    else:
        if random.random() >= prob:
            return x
        points = [[0, 0], [random.random(), random.random()], [random.random(), random.random()], [1, 1]]
    
    xpoints = sorted([p[0] for p in points])
    ypoints = sorted([p[1] for p in points])
    xvals, yvals = bezier_curve(points, nTimes=100000)
    '''if random.random() < 0.5:
        # Half change to get flip
        xvals = np.sort(xvals)
    else:
        xvals, yvals = np.sort(xvals), np.sort(yvals)'''
    xvals, yvals = np.sort(xvals), np.sort(yvals)
    nonlinear_x = np.interp(x, xvals, yvals)
    return nonlinear_x

def gamma_augmentation(x, range=(0.7, 1.5), random_state=None):
    if not isinstance(range, tuple):
        return x
    if isinstance(random_state, np.random.RandomState):
        gamma = random_state.uniform(*range)
        if random_state.random() < 0.5:
            gamma = 1/gamma
    
    else:
        gamma = np.random.uniform(*range)
        if np.random.random() < 0.5:
            gamma = 1/gamma
    
    return np.power(x, gamma)


def local_pixel_shuffling(x, prob=0.5, random_state=None):
    if isinstance(random_state, np.random.RandomState):
        if random_state.random() >= prob:
            return x
    else:
        if random.random() >= prob:
            return x
    
    image_temp = x.copy()#copy.deepcopy(x)
    orig_image = x.copy()#copy.deepcopy(x)
    _, img_deps, img_cols, img_rows = x.shape
    num_block = 10000
    for _ in range(num_block):
        if isinstance(random_state, np.random.RandomState):
            block_noise_size_x = random_state.randint(1, img_rows//16+1)
            block_noise_size_y = random_state.randint(1, img_cols//16+1)
            block_noise_size_z = random_state.randint(1, img_deps//16+1)
            noise_x = random_state.randint(0, img_rows-block_noise_size_x)
            noise_y = random_state.randint(0, img_cols-block_noise_size_y)
            noise_z = random_state.randint(0, img_deps-block_noise_size_z)
        else:
            block_noise_size_x = random.randint(1, img_rows//16)
            block_noise_size_y = random.randint(1, img_cols//16)
            block_noise_size_z = random.randint(1, img_deps//16)
            noise_x = random.randint(0, img_rows-block_noise_size_x)
            noise_y = random.randint(0, img_cols-block_noise_size_y)
            noise_z = random.randint(0, img_deps-block_noise_size_z)
        
        window = orig_image[0, noise_z:noise_z+block_noise_size_z, 
                               noise_y:noise_y+block_noise_size_y, 
                               noise_x:noise_x+block_noise_size_x,
                           ]
        window = window.flatten()

        if isinstance(random_state, np.random.RandomState):
            random_state.shuffle(window)
        else:
            np.random.shuffle(window)
        
        window = window.reshape((block_noise_size_z, 
                                 block_noise_size_y, 
                                 block_noise_size_x))
        image_temp[0, noise_z:noise_z+block_noise_size_z, 
                      noise_y:noise_y+block_noise_size_y, 
                      noise_x:noise_x+block_noise_size_x] = window
    local_shuffling_x = image_temp

    return local_shuffling_x

def image_in_painting(x, random_state=None):
    _, img_deps, img_cols, img_rows = x.shape
    cnt = 5
    
    if isinstance(random_state, np.random.RandomState):
        while cnt > 0 and random_state.random() < 0.95:
            block_noise_size_x = random_state.randint(img_rows//8, img_rows//4+1)
            block_noise_size_y = random_state.randint(img_cols//8, img_cols//4+1)
            block_noise_size_z = random_state.randint(img_deps//8, img_deps//4+1)
            noise_x = random_state.randint(3, img_rows-block_noise_size_x-2)
            noise_y = random_state.randint(3, img_cols-block_noise_size_y-2)
            noise_z = random_state.randint(3, img_deps-block_noise_size_z-2)
            x[:, 
            noise_z:noise_z+block_noise_size_z, 
            noise_y:noise_y+block_noise_size_y, 
            noise_x:noise_x+block_noise_size_x] = random_state.rand(block_noise_size_z, 
                                                                block_noise_size_y, 
                                                                block_noise_size_x)
    else:
        while cnt > 0 and random.random() < 0.95:
            block_noise_size_x = random.randint(img_rows//8, img_rows//4)
            block_noise_size_y = random.randint(img_cols//8, img_cols//4)
            block_noise_size_z = random.randint(img_deps//8, img_deps//4)
            noise_x = random.randint(3, img_rows-block_noise_size_x-3)
            noise_y = random.randint(3, img_cols-block_noise_size_y-3)
            noise_z = random.randint(3, img_deps-block_noise_size_z-3)
            x[:, 
            noise_z:noise_z+block_noise_size_z, 
            noise_y:noise_y+block_noise_size_y, 
            noise_x:noise_x+block_noise_size_x] = np.random.rand(block_noise_size_z, 
                                                                block_noise_size_y, 
                                                                block_noise_size_x)
    return x

def image_out_painting(x, random_state=None):
    #_, img_rows, img_cols, img_deps = x.shape
    _, img_deps, img_cols, img_rows = x.shape
    image_temp = x.copy()#copy.deepcopy(x)

    if isinstance(random_state, np.random.RandomState):
        x = random_state.rand(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        block_noise_size_x = img_rows - random_state.randint(img_rows//4, img_rows//2+1)
        block_noise_size_y = img_cols - random_state.randint(img_cols//4, img_cols//2+1)
        block_noise_size_z = img_deps - random_state.randint(img_deps//4, img_deps//2+1)
        noise_x = random_state.randint(3, img_rows-block_noise_size_x-2)
        noise_y = random_state.randint(3, img_cols-block_noise_size_y-2)
        noise_z = random_state.randint(3, img_deps-block_noise_size_z-2)
        x[:, 
        noise_z:noise_z+block_noise_size_z, 
        noise_y:noise_y+block_noise_size_y, 
        noise_x:noise_x+block_noise_size_x] = image_temp[:, noise_z:noise_z+block_noise_size_z, 
                                                        noise_y:noise_y+block_noise_size_y, 
                                                        noise_x:noise_x+block_noise_size_x]
        cnt = 4
        while cnt > 0 and random_state.random() < 0.95:
            block_noise_size_x = img_rows - random_state.randint(img_rows//4, img_rows//2+1)
            block_noise_size_y = img_cols - random_state.randint(img_cols//4, img_cols//2+1)
            block_noise_size_z = img_deps - random_state.randint(img_deps//4, img_deps//2+1)
            noise_x = random_state.randint(3, img_rows-block_noise_size_x-2)
            noise_y = random_state.randint(3, img_cols-block_noise_size_y-2)
            noise_z = random_state.randint(3, img_deps-block_noise_size_z-2)
            x[:, 
            noise_z:noise_z+block_noise_size_z, 
            noise_y:noise_y+block_noise_size_y, 
            noise_x:noise_x+block_noise_size_x] = image_temp[:, noise_z:noise_z+block_noise_size_z, 
                                                            noise_y:noise_y+block_noise_size_y, 
                                                            noise_x:noise_x+block_noise_size_x]
        '''x = np.random.rand(x.shape[0], x.shape[1], x.shape[2], x.shape[3], ) * 1.0
        block_noise_size_x = img_rows - random.randint(img_rows//4, img_rows//2)
        block_noise_size_y = img_cols - random.randint(img_cols//4, img_cols//2)
        block_noise_size_z = img_deps - random.randint(img_deps//4, img_deps//2)
        noise_x = random.randint(3, img_rows-block_noise_size_x-3)
        noise_y = random.randint(3, img_cols-block_noise_size_y-3)
        noise_z = random.randint(3, img_deps-block_noise_size_z-3)
        x[:, 
        noise_z:noise_z+block_noise_size_z, 
        noise_y:noise_y+block_noise_size_y, 
        noise_x:noise_x+block_noise_size_x] = image_temp[:, noise_z:noise_z+block_noise_size_z, 
                                                        noise_y:noise_y+block_noise_size_y, 
                                                        noise_x:noise_x+block_noise_size_x]
        cnt = 4
        while cnt > 0 and random.random() < 0.95:
            block_noise_size_x = img_rows - random.randint(img_rows//4, img_rows//2)
            block_noise_size_y = img_cols - random.randint(img_cols//4, img_cols//2)
            block_noise_size_z = img_deps - random.randint(img_deps//4, img_deps//2)
            noise_x = random.randint(3, img_rows-block_noise_size_x-3)
            noise_y = random.randint(3, img_cols-block_noise_size_y-3)
            noise_z = random.randint(3, img_deps-block_noise_size_z-3)
            x[:, 
            noise_z:noise_z+block_noise_size_z, 
            noise_y:noise_y+block_noise_size_y, 
            noise_x:noise_x+block_noise_size_x] = image_temp[:, noise_z:noise_z+block_noise_size_z, 
                                                            noise_y:noise_y+block_noise_size_y, 
                                                            noise_x:noise_x+block_noise_size_x]'''
    else:
        x = np.random.rand(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        block_noise_size_x = img_rows - random.randint(img_rows//4, img_rows//2)
        block_noise_size_y = img_cols - random.randint(img_cols//4, img_cols//2)
        block_noise_size_z = img_deps - random.randint(img_deps//4, img_deps//2)
        noise_x = random.randint(3, img_rows-block_noise_size_x-3)
        noise_y = random.randint(3, img_cols-block_noise_size_y-3)
        noise_z = random.randint(3, img_deps-block_noise_size_z-3)
        x[:, 
        noise_z:noise_z+block_noise_size_z, 
        noise_y:noise_y+block_noise_size_y, 
        noise_x:noise_x+block_noise_size_x] = image_temp[:, noise_z:noise_z+block_noise_size_z, 
                                                        noise_y:noise_y+block_noise_size_y, 
                                                        noise_x:noise_x+block_noise_size_x]
        cnt = 4
        while cnt > 0 and random.random() < 0.95:
            block_noise_size_x = img_rows - random.randint(img_rows//4, img_rows//2)
            block_noise_size_y = img_cols - random.randint(img_cols//4, img_cols//2)
            block_noise_size_z = img_deps - random.randint(img_deps//4, img_deps//2)
            noise_x = random.randint(3, img_rows-block_noise_size_x-3)
            noise_y = random.randint(3, img_cols-block_noise_size_y-3)
            noise_z = random.randint(3, img_deps-block_noise_size_z-3)
            x[:, 
            noise_z:noise_z+block_noise_size_z, 
            noise_y:noise_y+block_noise_size_y, 
            noise_x:noise_x+block_noise_size_x] = image_temp[:, noise_z:noise_z+block_noise_size_z, 
                                                            noise_y:noise_y+block_noise_size_y, 
                                                            noise_x:noise_x+block_noise_size_x]
    
    return x
                


def generate_pair(img, batch_size, config, status="test"):
    img_deps, img_cols, img_rows = img.shape[2], img.shape[3], img.shape[4]
    while True:
        index = [i for i in range(img.shape[0])]
        random.shuffle(index)
        y = img[index[:batch_size]]
        x = copy.deepcopy(y)
        for n in range(batch_size):
            
            # Autoencoder
            x[n] = copy.deepcopy(y[n])
            
            # Flip
            #x[n], y[n] = data_augmentation(x[n], y[n], config.flip_rate)

            # Local Shuffle Pixel
            x[n] = local_pixel_shuffling(x[n], prob=config.local_rate)
            
            # Apply non-Linear transformation with an assigned probability
            #x[n] = nonlinear_transformation(x[n], config.nonlinear_rate)
            
            # Inpainting & Outpainting
            if random.random() < config.paint_rate:
                if random.random() < config.inpaint_rate:
                    # Inpainting
                    x[n] = image_in_painting(x[n])
                else:
                    # Outpainting
                    x[n] = image_out_painting(x[n])

        # Save sample images module
        if config.save_samples is not None and status == "train" and random.random() < 0.01:
            print('Saving samples')
            n_sample = random.choice( [i for i in range(config.batch_size)] )
            sample_1 = np.concatenate((x[n_sample,0,2*img_deps//6,:,:], y[n_sample,0,2*img_deps//6,:,:]), axis=0)
            sample_2 = np.concatenate((x[n_sample,0,3*img_deps//6,:,:], y[n_sample,0,3*img_deps//6,:,:]), axis=0)
            sample_3 = np.concatenate((x[n_sample,0,4*img_deps//6,:,:], y[n_sample,0,4*img_deps//6,:,:]), axis=0)
            sample_4 = np.concatenate((x[n_sample,0,5*img_deps//6,:,:], y[n_sample,0,5*img_deps//6,:,:]), axis=0)
            final_sample = np.concatenate((sample_1, sample_2, sample_3, sample_4), axis=1)
            final_sample = final_sample * 255.0
            final_sample = final_sample.astype(np.uint8)
            file_name = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(10)])+'.'+config.save_samples
            cv2.imwrite(os.path.join(config.sample_path, config.exp_name, file_name), final_sample)
            #imageio.imwrite(os.path.join(config.sample_path, config.exp_name, file_name), final_sample)

        yield (x, y)