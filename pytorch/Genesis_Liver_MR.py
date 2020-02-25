#!/usr/bin/env python
# coding: utf-8


import warnings
warnings.filterwarnings('ignore')
import numpy as np
from torch import nn
import torch
from torchsummary import summary
import sys
from utils import *
from npydataset import NpyDataset
from torch.utils.data import DataLoader
import unet3d
from config import models_genesis_config_mr
from tqdm import tqdm

print("torch = {}".format(torch.__version__))


#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

conf = models_genesis_config_mr()
conf.display()

'''x_train = []
for i,fold in enumerate(tqdm(conf.train_fold)):
    file_name = "bat_"+str(conf.scale)+"_"+str(conf.input_rows)+"x"+str(conf.input_cols)+"x"+str(conf.input_deps)+"_"+str(fold)+".npy"
    s = np.load(os.path.join(conf.data, 'train', file_name))
    x_train.extend(s)
x_train = np.expand_dims(np.array(x_train), axis=1)

x_valid = []
for i,fold in enumerate(tqdm(conf.valid_fold)):
    file_name = "bat_"+str(conf.scale)+"_"+str(conf.input_rows)+"x"+str(conf.input_cols)+"x"+str(conf.input_deps)+"_"+str(fold)+".npy"
    s = np.load(os.path.join(conf.data, 'val', file_name))
    x_valid.extend(s)
x_valid = np.expand_dims(np.array(x_valid), axis=1)

print("x_train: {} | {:.2f} ~ {:.2f}".format(x_train.shape, np.min(x_train), np.max(x_train)))
print("x_valid: {} | {:.2f} ~ {:.2f}".format(x_valid.shape, np.min(x_valid), np.max(x_valid)))

training_generator = generate_pair(x_train,conf.batch_size, conf, status='train')
validation_generator = generate_pair(x_valid,conf.batch_size, conf)'''

train_dataset = NpyDataset(os.path.join(conf.data, 'train'), conf.train_fold, train=True)
valid_dataset = NpyDataset(os.path.join(conf.data, 'val'), conf.valid_fold, train=False)
train_loader = DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True, num_workers=6)
valid_loader = DataLoader(valid_dataset, batch_size=conf.batch_size, num_workers=6)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = unet3d.UNet3D()
model = nn.DataParallel(model, device_ids = [i for i in range(torch.cuda.device_count())])
model.to(device)

for m in model.modules():
	if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
		nn.init.kaiming_normal_(m.weight)
	elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm):
		nn.init.constant_(m.weight, 1)
		nn.init.constant_(m.bias, 0)
	elif isinstance(m, nn.Linear):
		nn.init.constant_(m.bias, 0)

print("Total CUDA devices: ", torch.cuda.device_count())

summary(model, (1, conf.input_deps, conf.input_cols, conf.input_rows), batch_size=conf.batch_size)
#criterion = nn.MSELoss()
criterion = nn.L1Loss()

if conf.optimizer == "sgd":
	optimizer = torch.optim.SGD(model.parameters(), conf.lr, momentum=0.9, weight_decay=0.0001, nesterov=True)
elif conf.optimizer == "adam":
	optimizer = torch.optim.Adam(model.parameters(), conf.lr)
else:
	raise

#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(conf.patience * 0.8), gamma=0.5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

# to track the training loss as the model trains
train_losses = []
# to track the validation loss as the model trains
valid_losses = []
# to track the average training loss per epoch as the model trains
avg_train_losses = []
# to track the average validation loss per epoch as the model trains
avg_valid_losses = []
best_loss = 100000
intial_epoch =0
num_epoch_no_improvement = 0
sys.stdout.flush()

if conf.weights != None:
	checkpoint=torch.load(conf.weights)
	model.load_state_dict(checkpoint['state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	intial_epoch=checkpoint['epoch']
	best_loss = checkpoint['best_loss']
	print("Loading weights from ",conf.weights)
elif conf.pretrained != None:
	checkpoint=torch.load(conf.pretrained)
	model.load_state_dict(checkpoint['state_dict'])
	#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	#intial_epoch=checkpoint['epoch']
	#best_loss = checkpoint['best_loss']
	print("Loading init weights from ",conf.pretrained)

sys.stdout.flush()

if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
	scheduler.step(best_loss)

for epoch in range(intial_epoch,conf.nb_epoch):
	if isinstance(scheduler, torch.optim.lr_scheduler.StepLR):
		scheduler.step(epoch)
	model.train()
	print('Learning rate: %f' % (scheduler._last_lr[0]))
	#for iteration in range(int(x_train.shape[0]//conf.batch_size)):
	for iteration, (image, gt) in enumerate(train_loader):
		image,gt = image.to(device), gt.to(device)
		#gt = np.repeat(gt,conf.nb_class,axis=1)
		gt = gt.repeat(1, conf.nb_class, 1, 1, 1)
		pred=model(image)
		loss = criterion(pred,gt)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		train_losses.append(round(loss.item(), 2))
		if (iteration + 1) % 5 ==0:
			print('Epoch [{}/{}], iteration {}, Loss: {:.6f}'
				.format(epoch + 1, conf.nb_epoch, iteration + 1, np.average(train_losses)))
			sys.stdout.flush()
			if (iteration + 1) % 200 == 0:
				x = image[0].cpu().numpy()
				y = gt[0].cpu().numpy()
				p = pred[0].detach().cpu().numpy()
				sample_1 = np.concatenate((x[0,2*x.shape[1]//6,:,:], y[0,2*x.shape[1]//6,:,:], p[0,2*x.shape[1]//6,:,:]), axis=0)
				sample_2 = np.concatenate((x[0,3*x.shape[1]//6,:,:], y[0,3*x.shape[1]//6,:,:], p[0,3*x.shape[1]//6,:,:]), axis=0)
				sample_3 = np.concatenate((x[0,4*x.shape[1]//6,:,:], y[0,4*x.shape[1]//6,:,:], p[0,4*x.shape[1]//6,:,:]), axis=0)
				sample_4 = np.concatenate((x[0,5*x.shape[1]//6,:,:], y[0,5*x.shape[1]//6,:,:], p[0,5*x.shape[1]//6,:,:]), axis=0)
				final_sample = np.concatenate((sample_1, sample_2, sample_3, sample_4), axis=1)
				final_sample = final_sample * 255.0
				final_sample = final_sample.astype(np.uint8)
				file_name = str(epoch+1)+'_'+''.join([random.choice(string.ascii_letters + string.digits) for n in range(10)])+'.png'
				cv2.imwrite(os.path.join('sample', 'train', file_name), final_sample)

	with torch.no_grad():
		model.eval()
		print("validating....")
		#for i in range(int(x_valid.shape[0]//conf.batch_size)):
		for i, (image,gt) in enumerate(valid_loader):
			image=image.to(device)
			gt=gt.to(device)
			pred=model(image)
			loss = criterion(pred,gt)
			valid_losses.append(loss.item())

			for j in range(image.shape[0]):
				if random.random() < 0.01:
					x = image[j].cpu().numpy()
					y = gt[j].cpu().numpy()
					p = pred[j].cpu().numpy()
					sample_1 = np.concatenate((x[0,2*x.shape[1]//6,:,:], y[0,2*x.shape[1]//6,:,:], p[0,2*x.shape[1]//6,:,:]), axis=0)
					sample_2 = np.concatenate((x[0,3*x.shape[1]//6,:,:], y[0,3*x.shape[1]//6,:,:], p[0,3*x.shape[1]//6,:,:]), axis=0)
					sample_3 = np.concatenate((x[0,4*x.shape[1]//6,:,:], y[0,4*x.shape[1]//6,:,:], p[0,4*x.shape[1]//6,:,:]), axis=0)
					sample_4 = np.concatenate((x[0,5*x.shape[1]//6,:,:], y[0,5*x.shape[1]//6,:,:], p[0,5*x.shape[1]//6,:,:]), axis=0)
					final_sample = np.concatenate((sample_1, sample_2, sample_3, sample_4), axis=1)
					final_sample = final_sample * 255.0
					final_sample = final_sample.astype(np.uint8)
					file_name = str(epoch+1)+'_'+''.join([random.choice(string.ascii_letters + string.digits) for n in range(10)])+'.png'
					cv2.imwrite(os.path.join('sample', 'val', file_name), final_sample)

			if (i + 1) % 10 == 0:
				print('Epoch [{}/{}], iteration {}, Loss: {:.6f}'
				.format(epoch + 1, conf.nb_epoch, i + 1, np.average(valid_losses)))
				sys.stdout.flush()
				x = image[0].cpu().numpy()
				y = gt[0].cpu().numpy()
				p = pred[0].cpu().numpy()
				sample_1 = np.concatenate((x[0,2*x.shape[1]//6,:,:], y[0,2*x.shape[1]//6,:,:], p[0,2*x.shape[1]//6,:,:]), axis=0)
				sample_2 = np.concatenate((x[0,3*x.shape[1]//6,:,:], y[0,3*x.shape[1]//6,:,:], p[0,3*x.shape[1]//6,:,:]), axis=0)
				sample_3 = np.concatenate((x[0,4*x.shape[1]//6,:,:], y[0,4*x.shape[1]//6,:,:], p[0,4*x.shape[1]//6,:,:]), axis=0)
				sample_4 = np.concatenate((x[0,5*x.shape[1]//6,:,:], y[0,5*x.shape[1]//6,:,:], p[0,5*x.shape[1]//6,:,:]), axis=0)
				final_sample = np.concatenate((sample_1, sample_2, sample_3, sample_4), axis=1)
				final_sample = final_sample * 255.0
				final_sample = final_sample.astype(np.uint8)
				file_name = str(epoch+1)+'_'+''.join([random.choice(string.ascii_letters + string.digits) for n in range(10)])+'.png'
				cv2.imwrite(os.path.join('sample', 'val', file_name), final_sample)
	
	#logging
	train_loss=np.average(train_losses)
	valid_loss=np.average(valid_losses)

	if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
		scheduler.step(valid_loss)
	
	avg_train_losses.append(train_loss)
	avg_valid_losses.append(valid_loss)
	print("Epoch {}, validation loss is {:.4f}, training loss is {:.4f}".format(epoch+1,valid_loss,train_loss))
	train_losses=[]
	valid_losses=[]
	if valid_loss < best_loss:
		print("Validation loss decreases from {:.4f} to {:.4f}".format(best_loss, valid_loss))
		best_loss = valid_loss
		num_epoch_no_improvement = 0
		#save model
		torch.save({
			'epoch': epoch+1,
			'best_loss': best_loss,
			'loss': valid_loss,
			'state_dict' : model.state_dict(),
			'optimizer_state_dict': optimizer.state_dict()
		},os.path.join(conf.model_path, "Genesis_Liver_MR.pt"))
		print("Saving model ",os.path.join(conf.model_path,"Genesis_Liver_MR.pt"))
	else:
		print("Validation loss does not decrease from {:.4f}, num_epoch_no_improvement {}".format(best_loss,num_epoch_no_improvement))
		num_epoch_no_improvement += 1
	torch.save({
		'epoch': epoch+1,
		'best_loss': best_loss,
		'loss': valid_loss,
		'state_dict' : model.state_dict(),
		'optimizer_state_dict': optimizer.state_dict()
	},os.path.join(conf.model_path, "epoch_%03d.pt" % (epoch+1)))
	if num_epoch_no_improvement == conf.patience:
		print("Early Stopping")
		break
	sys.stdout.flush()
