import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import Optimizer
import torch.backends.cudnn as cudnn
import torchvision
from torch.utils.data import TensorDataset, DataLoader
import os, logging, sys
import random
import matplotlib.pyplot as plt
import numpy as np
import hypergrad as hg
from itertools import repeat
from torchvision.datasets import CIFAR10
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import tqdm

from poi_util import poison_dataset,patching_test
import poi_util
from models import *


root = './datasets/'

class Tee(object):
	def __init__(self, name, mode):
		self.file = open(name, mode)
		self.stdout = sys.stdout
		sys.stdout = self
	def __del__(self):
		sys.stdout = self.stdout
		self.file.close()
	def write(self, data):
		if not '...' in data:
			self.file.write(data)
		self.stdout.write(data)
		self.flush()
	def flush(self):
		self.file.flush()

def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger

def get_results(model, criterion, data_loader, device='cuda'):
	model.eval()
	val_loss = 0
	correct = 0
	total = 0
	with torch.no_grad():
		for batch_idx, (inputs, targets) in enumerate(data_loader):
			inputs, targets = inputs.to(device), targets.to(device)
			outputs = model(inputs)
			loss = criterion(outputs, targets.long())

			val_loss += loss.item()
			_, predicted = outputs.max(1)
			total += targets.size(0)
			correct += predicted.eq(targets).sum().item()
		return correct / total

# def get_eval_data(name, attack_name='badnets', target_lab='8', args=None):
# 	if name == 'cifar10':
# 		testset = CIFAR10(root, train=False, transform=None, download=True)
# 		x_test, y_test = testset.data, testset.targets
# 		x_test = x_test.astype('float32')/255
# 		y_test = np.asarray(y_test)

# 		x_poi_test,y_poi_test= patching_test(x_test, y_test, attack_name, target_lab=target_lab)

# 		y_test = torch.Tensor(y_test.reshape((-1,)).astype(np.int))
# 		y_poi_test = torch.Tensor(y_poi_test.reshape((-1,)).astype(np.int))

# 		x_test = torch.Tensor(np.transpose(x_test,(0,3,1,2)))
# 		x_poi_test = torch.Tensor(np.transpose(x_poi_test,(0,3,1,2)))
	
# 	elif name == 'gtsrb':
# 		import h5py 
# 		f = h5py.File('./datasets/gtsrb_dataset.h5','r') 
# 		x_train = np.asarray(f['X_train'])/255
# 		x_test = np.asarray(f['X_test'])/255
# 		y_train = np.argmax(np.asarray(f['Y_train']),axis=1)
# 		y_test = np.argmax(np.asarray(f['Y_test']),axis=1)

# 		randidx = np.arange(y_test.shape[0])
# 		np.random.shuffle(randidx)
# 		x_test = x_test[randidx]
# 		y_test = y_test[randidx]

# 		x_poi_test,y_poi_test= patching_test(x_test, y_test, attack_name, target_lab=target_lab)

# 		y_test = torch.Tensor(y_test.reshape((-1,)).astype(np.int))
# 		y_poi_test = torch.Tensor(y_poi_test.reshape((-1,)).astype(np.int))

# 		x_test = torch.Tensor(np.transpose(x_test,(0,3,1,2)))
# 		x_poi_test = torch.Tensor(np.transpose(x_poi_test,(0,3,1,2)))

# 	test_set = TensorDataset(x_test[5000:],y_test[5000:])
# 	att_val_set = TensorDataset(x_poi_test[:5000],y_poi_test[:5000])
# 	if args.unl_set == None:
# 		unl_set = TensorDataset(x_test[:5000],y_test[:5000])
# 	else:
# 		unl_set = args.unl_set

# 	return test_set, att_val_set, unl_set

import sys
sys.path.append('../../')
from tools.utils import manual_seed, get_model, rm_if_exist
from tools.dataset import get_train_and_test_dataset, PartialDataset, PoisonDataset, get_dataset_class_and_scale
from torch.utils.data.dataloader import DataLoader
import os
from omegaconf import OmegaConf
if __name__ == "__main__":
	global args, logger

	parser = ArgumentParser(description='I-BAU defense')
	parser.add_argument('--dataset', default='cifar10', help='the dataset to use')
	parser.add_argument('--poi_path', default='./checkpoint/badnets_8_02_ckpt.pth', help='path of the poison model need to be unlearn')
	parser.add_argument('--log_path', default='./unlearn_logs', help='path of the log file')
	parser.add_argument('--device', type=str, default='1', help='Device to use. Like cuda, cuda:0 or cpu')
	parser.add_argument('--batch_size', type=int, default=128, help='batch size of unlearn loader')
	parser.add_argument('--unl_set', default=None, help='extra unlearn dataset, if None then use test data')
	parser.add_argument('--optim', type=str, default='SGD', help='SDG/Adam')
	parser.add_argument('--lr', default=1e-3, type=float, help='learning rate of outer loop optimizer')

	## hyper params
	parser.add_argument('--n_rounds', default=200, type=int, help='the maximum number of unelarning rounds')
	parser.add_argument('--K', default=5, type=int, help='the maximum number of fixed point iterations')

	parser.add_argument('--path', type=str, default='/home/chengyiqiu/code/INBA/results/cifar10/badnet/fip/20241225205432', help='the results folder path.')
	


	args = parser.parse_args()
	args.poi_path = f'{args.path}/results.pth'
	args.log_path = f'{args.path}/IBAU'
	rm_if_exist(args.log_path)
	os.makedirs(args.log_path, exist_ok=True)

	logger = get_logger()
	logger.info(args)
	logger.info("=> Setup defense..")

	os.makedirs(args.log_path, exist_ok=True)
	log_file = "{}.txt".format(args.dataset)
	Tee(os.path.join(args.log_path, log_file), 'w')

	device = f'cuda:{args.device}'

	logger.info('==> Preparing data..')
	path = f'{args.path}/config.yaml'
	config = OmegaConf.load(path)
	config.attack.mode = 'defense'
	manual_seed(config.seed)

	train_ds, test_set = get_train_and_test_dataset(config.dataset_name)
	unl_set = PartialDataset(test_set, 0.5)
	poison_config = config.copy()
	poison_config.ratio = 1
	att_val_set = PoisonDataset(unl_set, poison_config)
	bs = args.batch_size
	#data loader for verifying the clean test accuracy
	clnloader = torch.utils.data.DataLoader(
		test_set, batch_size=bs, shuffle=False, num_workers=2)

	#data loader for verifying the attack success rate
	poiloader_cln = torch.utils.data.DataLoader(
		unl_set, batch_size=bs, shuffle=False, num_workers=2)

	poiloader = torch.utils.data.DataLoader(
		att_val_set, batch_size=bs, shuffle=False, num_workers=2)

	#data loader for the unlearning step
	unlloader = torch.utils.data.DataLoader(
		unl_set, batch_size=bs, shuffle=False, num_workers=2)


	### initialize theta
	num_classes, _ = get_dataset_class_and_scale(config.dataset_name)
	model = get_model(config.model, num_classes, device)
	criterion = nn.CrossEntropyLoss()
	# model.load_state_dict(torch.load(args.poi_path)['net'])
	model = get_model(config.model, num_classes, device=device)
	ld = torch.load(f'{args.path}/results.pth', map_location=device)
	model.load_state_dict(ld['model'])
	model.to(device)
	model.eval()
	if args.optim == 'SGD':
		outer_opt = torch.optim.SGD(model.parameters(), lr=args.lr)
	elif args.optim == 'Adam':
		outer_opt = torch.optim.Adam(model.parameters(), lr=args.lr)

	ACC = get_results(model, criterion, clnloader, device)
	ASR = get_results(model, criterion, poiloader, device)
	print('Original ACC:', ACC)
	print('Original ASR:', ASR)

	### define the inner loss L2
	def loss_inner(perturb, model_params):
		images = images_list[0].to(device)
		labels = labels_list[0].long().to(device)
	#     per_img = torch.clamp(images+perturb[0],min=0,max=1)
		per_img = images+perturb[0]
		per_logits = model.forward(per_img)
		loss = F.cross_entropy(per_logits, labels, reduction='none')
		loss_regu = torch.mean(-loss) +0.001*torch.pow(torch.norm(perturb[0]),2)
		return loss_regu

	### define the outer loss L1
	def loss_outer(perturb, model_params):
		portion = 0.01
		images, labels = images_list[batchnum].to(device), labels_list[batchnum].long().to(device)
		patching = torch.zeros_like(images, device=device)
		number = images.shape[0]
		rand_idx = random.sample(list(np.arange(number)),int(number*portion))
		patching[rand_idx] = perturb[0]
	#     unlearn_imgs = torch.clamp(images+patching,min=0,max=1)
		unlearn_imgs = images+patching
		logits = model(unlearn_imgs)
		criterion = nn.CrossEntropyLoss()
		loss = criterion(logits, labels)
		return loss

	images_list, labels_list = [], []
	for index, (images, labels) in enumerate(unlloader):
		images_list.append(images)
		labels_list.append(labels)
	inner_opt = hg.GradientDescent(loss_inner, 0.1)


	### inner loop and optimization by batch computing
	logger.info("=> Conducting Defence..")
	ASR_list = [get_results(model, criterion, poiloader, device)]
	ACC_list = [get_results(model, criterion, clnloader, device)]

	# for param in model.parameters():
	# 	param.requires_grad = True

	for round in range(args.n_rounds):
		# batch_pert = torch.zeros_like(test_set.tensors[0][:1], requires_grad=True, device='cuda')
		batch_pert = torch.zeros((3, 32 ,32), requires_grad=True, device=device)
		batch_opt = torch.optim.SGD(params=[batch_pert],lr=10)
	
		for images, labels in unlloader:
			images = images.to(device)
			ori_lab = torch.argmax(model.forward(images),axis = 1).long()
	#         per_logits = model.forward(torch.clamp(images+batch_pert,min=0,max=1))
			per_logits = model.forward(images+batch_pert)
			loss = F.cross_entropy(per_logits, ori_lab, reduction='mean')
			loss_regu = torch.mean(-loss) +0.001*torch.pow(torch.norm(batch_pert),2)
			batch_opt.zero_grad()
			loss_regu.backward(retain_graph=True)
			batch_opt.step()
		#l2-ball
		# pert = batch_pert * min(1, 10 / torch.norm(batch_pert))
		pert = batch_pert
		#unlearn step
		for batchnum in range(len(images_list)): 
			outer_opt.zero_grad()
			hg.fixed_point(pert, list(model.parameters()), args.K, inner_opt, loss_outer) 
			outer_opt.step()

		ASR_list.append(get_results(model,criterion,poiloader,device))
		ACC_list.append(get_results(model,criterion,clnloader,device))
		print('Round:',round)
		
		print('ACC:',get_results(model,criterion,clnloader,device))
		print('ASR:',get_results(model,criterion,poiloader,device))
	res = {
		'acc_list': ACC_list,
		'asr_list': ASR_list,
	}
	torch.save(res, f'{args.log_path}/plot_results.pth')

	
