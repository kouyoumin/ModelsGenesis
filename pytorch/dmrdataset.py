import torch
from torch.utils.data import Dataset
import numpy as np
from skimage import transform
import os
from DynamicMRI import DynamicMRI
from utils import gamma_augmentation


class DMRDataset(Dataset):
    def __init__(self, root_path, folds, sizes=(248, 256, 264, 272, 280, 288), crop=128, crop_z=32, train=True):
        self.root_path = root_path
        self.folds = []
        self.train = train
        self.crop = crop
        self.crop_z = crop_z
        self.imgs = []
        self.annos = []

        for fold in folds:
            subset_path = os.path.join(root_path, '%02d' % (fold))
            dmr = DynamicMRI(subset_path, annotated_only=True)
            if len(dmr.phases) > 0:
                phase = dmr.phases[-1]
                for size in sizes:
                    np_img = np.expand_dims(phase.restore_original().resize(size, phase.image.GetHeight() * size // phase.image.GetWidth(), phase.image.GetDepth()).numpy().astype(np.float32), axis=0)
                    np_img = (np_img-np_img.min()) / (np_img.max()-np_img.min())
                    self.imgs.append(np_img)
                    #print(dmr.phases[-1].annotation.shape)
                    self.annos.append(transform.resize(dmr.phases[-1].annotation, (8, phase.image.GetDepth(), phase.image.GetHeight() * size // phase.image.GetWidth(), size), order=1))
                    self.folds.append(fold)
                    #print(self.imgs[-1].shape, self.annos[-1].shape)
                    assert(self.imgs[-1].shape[1:] == self.annos[-1].shape[1:])
                    assert(self.annos[-1].min() == 0)
                    #print(dmr.phases[-1].annotation.max(), self.annos[-1].max())
                    assert(self.annos[-1].max() == 1)
                    print('%02d: %s %s %s' % (fold, phase.get_desc(), phase.orig_image.GetSize(), str(self.imgs[-1].shape)))
                    for i in range(8):
                        print('Mask %d: max=%d, min=%d' % (i, self.annos[-1][i].max(), self.annos[-1][i].min()))


    def __getitem__(self, idx):
        if self.crop == 0:
            return torch.from_numpy(self.imgs[idx][:, :, :, :]).float(), torch.from_numpy(self.annos[idx][:, :, :, :]).float()
            
        #print(self.folds[idx], self.imgs[idx].shape)
        if self.train:            
            #Random crop
            start_x = np.random.randint(0, self.imgs[idx].shape[3] - self.crop + 1)
            start_y = np.random.randint(0, self.imgs[idx].shape[2] - self.crop + 1)
            if self.crop_z == 0:
                return torch.from_numpy(gamma_augmentation(self.imgs[idx][:, :, start_y:start_y+self.crop, start_x:start_x+self.crop])).float(), torch.from_numpy(self.annos[idx][:, :, start_y:start_y+self.crop, start_x:start_x+self.crop]).float()
            else:
                start_z = np.random.randint(0, self.imgs[idx].shape[1] - self.crop_z + 1)
                return torch.from_numpy(gamma_augmentation(self.imgs[idx][:, start_z:start_z+self.crop_z, start_y:start_y+self.crop, start_x:start_x+self.crop])).float(), torch.from_numpy(self.annos[idx][:, start_z:start_z+self.crop_z, start_y:start_y+self.crop, start_x:start_x+self.crop]).float()
        else:            
            rs = np.random.RandomState(seed=idx)
            #Random crop
            start_x = rs.randint(0, self.imgs[idx].shape[3] - self.crop + 1)
            start_y = rs.randint(0, self.imgs[idx].shape[2] - self.crop + 1)
            if self.crop_z == 0:
                return torch.from_numpy(self.imgs[idx][:, :, start_y:start_y+self.crop, start_x:start_x+self.crop]).float(), torch.from_numpy(self.annos[idx][:, :, start_y:start_y+self.crop, start_x:start_x+self.crop]).float()
            else:
                start_z = rs.randint(0, self.imgs[idx].shape[1] - self.crop_z + 1)
                return torch.from_numpy(self.imgs[idx][:, start_z:start_z+self.crop_z, start_y:start_y+self.crop, start_x:start_x+self.crop]).float(), torch.from_numpy(self.annos[idx][:, start_z:start_z+self.crop_z, start_y:start_y+self.crop, start_x:start_x+self.crop]).float()
        

    def __len__(self):
        return len(self.imgs)


def visualize(dataset):
    import cv2
    for idx in range(len(dataset)):
        image, gt = dataset[idx]
        gt=gt[2:, :, :, :]

        x = image[0].cpu().numpy()
        y = (gt[0].cpu()>0.5).float().numpy() * (2**2)/255
        for j in range(1,6):
            y += (gt[j].cpu()>0.5).float().numpy() * (2**(j+2))/255
        
        samples = []
        #print(x.shape, y.shape)
        for i in range(image.size()[1]):
            samples.append(np.concatenate((x[i,:,:], y[i,:,:]), axis=0))
        final_sample = np.concatenate(samples, axis=1)
        final_sample = final_sample * 255.0
        final_sample = final_sample.astype(np.uint8)
        file_name = '%02d.png' % (dataset.folds[idx])
        if not os.path.isdir(os.path.join(dataset.root_path, '%02d'%(dataset.folds[idx]), 'vis')):
            os.makedirs(os.path.join(dataset.root_path, '%02d'%(dataset.folds[idx]), 'vis'))
        cv2.imwrite(os.path.join(dataset.root_path, '%02d'%(dataset.folds[idx]), 'vis', file_name), final_sample)


if __name__ == '__main__':
    import sys
    import time
    import pickle
    #dmri = DMRDataset(sys.argv[1], (1,2,3,4,5,6,7,8,9,10,11,12,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64), crop=0, train=False)
    dmri = DMRDataset(sys.argv[1], (65,66,67,68,69,70), crop=0, train=False)
    visualize(dmri)
    timestamp = time.time()
    with open(os.path.join(sys.argv[1], '%s.pickle'%(str(timestamp))), 'wb') as file:
        pickle.dump(dmri, file)
        file.close()
