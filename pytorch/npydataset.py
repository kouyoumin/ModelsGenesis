import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm

from utils import *

class NpyDataset(Dataset):
    def __init__(self, root_path, folds, scale=32, input_rows=128, input_cols=128, input_deps=32, local_rate=0.5, paint_rate=0.9, inpaint_rate=0.2, train=True):
        self.local_rate = local_rate
        self.paint_rate = paint_rate
        self.inpaint_rate = inpaint_rate
        self.train = train

        arrays = []
        for i,fold in enumerate(tqdm(folds)):
            file_name = "bat_"+str(scale)+"_"+str(input_rows)+"x"+str(input_cols)+"x"+str(input_deps)+"_"+str(fold)+".npy"
            s = np.load(os.path.join(root_path, file_name))
            arrays.extend(s)
        self.array = np.expand_dims(np.array(arrays), axis=1)
        print("Dataset: {} | {:.2f} ~ {:.2f}".format(self.array.shape, np.min(self.array), np.max(self.array)))
    
    def __getitem__(self, idx):
        y = self.array[idx]
        
        if self.train:
            x = copy.deepcopy(y)
            
            # Flip
            #x[n], y[n] = data_augmentation(x[n], y[n], config.flip_rate)

            # Local Shuffle Pixel
            x = local_pixel_shuffling(x, prob=self.local_rate)
            
            # Apply non-Linear transformation with an assigned probability
            x = nonlinear_transformation(x)
            
            # Inpainting & Outpainting
            if random.random() < self.paint_rate:
                if random.random() < self.inpaint_rate:
                    # Inpainting
                    x = image_in_painting(x)
                else:
                    # Outpainting
                    x = image_out_painting(x)
            
            if random.random() < 0.001:
                print('Saving samples')
                sample_1 = np.concatenate((x[0,2*x.shape[1]//6,:,:], y[0,2*x.shape[1]//6,:,:]), axis=0)
                sample_2 = np.concatenate((x[0,3*x.shape[1]//6,:,:], y[0,3*x.shape[1]//6,:,:]), axis=0)
                sample_3 = np.concatenate((x[0,4*x.shape[1]//6,:,:], y[0,4*x.shape[1]//6,:,:]), axis=0)
                sample_4 = np.concatenate((x[0,5*x.shape[1]//6,:,:], y[0,5*x.shape[1]//6,:,:]), axis=0)
                final_sample = np.concatenate((sample_1, sample_2, sample_3, sample_4), axis=1)
                final_sample = final_sample * 255.0
                final_sample = final_sample.astype(np.uint8)
                file_name = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(10)])+'.png'
                cv2.imwrite(os.path.join('sample', 'train', file_name), final_sample)
            
            return torch.FloatTensor(x), torch.FloatTensor(y)
        else:
            return torch.FloatTensor(y), torch.FloatTensor(y)
        
        

    def __len__(self):
        return self.array.shape[0]