#!/usr/bin/env python
# coding: utf-8


import warnings
warnings.filterwarnings('ignore')
import numpy as np
from torch import nn
import torch.nn.functional as F
import torch
from torchsummary import summary
import sys
from utils import *
from dmrdataset import DMRDataset
from torch.utils.data import DataLoader
import unet3d
from testconfig import models_genesis_config_mr2_test, models_genesis_config_mr6_test
from dictmod import remove_module_prefix

print("torch = {}".format(torch.__version__))

if '2' in sys.argv:
	conf = models_genesis_config_mr2_test()
else:
	conf = models_genesis_config_mr6_test()

if not os.path.isdir('result'):
	os.makedirs('result')

test_dataset = DMRDataset(os.path.join(conf.data), conf.test_fold, sizes=(256,), crop=0, train=False)
test_loader = DataLoader(test_dataset, batch_size=1, num_workers=2)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = unet3d.UNet3D(in_channel=1, out_channel=conf.nb_class, sigmoid=True)

criterion_val = nn.BCELoss().to(device)

test_losses = []

if conf.weights != None:
	checkpoint=torch.load(conf.weights, map_location=torch.device('cpu'))
	model.load_state_dict(remove_module_prefix(checkpoint['state_dict']))
	print("Load from",conf.weights)
else:
	print('Error: no weight specified')
	exit(1)

model.to(device)
sys.stdout.flush()

with torch.no_grad():
	model.eval()
	print("Testing....")
	#for i in range(int(x_valid.shape[0]//conf.batch_size)):
	for i, (image,gt) in enumerate(test_loader):
		if conf.nb_class == 2:
			target=gt[:, 6:, :, :, :].to(device)
			target[:,0,:,:,:] = torch.sum(gt[:, 2:-1, :, :, :], 1, keepdim=False).clamp_(0.0,1.0)
		else:
			target = gt[:, 8-conf.nb_class:, :, :, :]
		image,target = image.to(device), target.to(device)
		pred=model(image)
		loss = criterion_val(pred, target)
		test_losses.append(loss.item())
		if (i + 1) % 1 == 0:
			print('iteration {}, Loss: {:.6f}'
			.format(i + 1, loss.item()))
			sys.stdout.flush()
			x = image[0].cpu().numpy()
			#y = gt[0].cpu().numpy()
			y = (target[0][0].cpu()>0.5).float().numpy() * (2**(0+(8-conf.nb_class)))/255
			#for j in range(1,6):
			for j in range(1, conf.nb_class):
				y += (target[0][j].cpu()>0.5).float().numpy() * (2**(j+(8-conf.nb_class)))/255
			
			#print(pred[0][0].max())
			p = (pred[0][0].cpu()>0.5).float().numpy() * (2**(0+(8-conf.nb_class)))/255
			for j in range(1, conf.nb_class):
				#p += (pred[0][j].cpu()>0.5).float().numpy() * (2**(j+2))/255
				p += (pred[0][j].cpu()>0.5).float().numpy() * (2**(j+(8-conf.nb_class)))/255
			
			samples = []
			for k in range(image.size()[2]):
				samples.append(np.concatenate((x[0,k,:,:], y[k,:,:], p[k,:,:]), axis=0))
			final_sample = np.concatenate(samples, axis=1)
			final_sample = final_sample * 255.0
			final_sample = final_sample.astype(np.uint8)
			file_name = 'test_'+str(i+1)+'.png'
			cv2.imwrite(os.path.join('result', file_name), final_sample)

	test_loss=np.average(test_losses)
	print('Test loss: %f' % (test_loss))
	sys.stdout.flush()
