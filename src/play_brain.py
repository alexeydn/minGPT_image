from torchvision import datasets, transforms
import torch
import numpy as np
import torchvision
from read_dataset import BrainDataset
from mingpt.model import GPT, GPTConfig
from mingpt.trainer import Trainer, TrainerConfig
from mingpt.utils import sample
import os
from mingpt.utils import set_seed
import cv2
import logging


logging.basicConfig(filename='training_info.log', encoding='utf-8', level=logging.DEBUG)
set_seed(42)

# read datasets
perm = torch.randperm(32*32)

training_path = os.path.join(os.getcwd(), '../data/training')
train_dataset = BrainDataset(training_path, perm)

test_path = os.path.join(os.getcwd(), '../data/test')
test_dataset = BrainDataset(test_path, perm)

# construct a GPT model
mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
                  embd_pdrop=0, resid_pdrop=0, attn_pdrop=0,
                  n_layer=4, n_head=4, n_embd=256)
model = GPT(mconf)
#checkpoint = torch.load('brain_model_4l_4h.pt')
#model.load_state_dict(checkpoint)

tokens_per_epoch = len(train_dataset) * train_dataset.block_size
train_epochs = 30

# initialize a trainer instance and kick off training
tconf = TrainerConfig(max_epochs=train_epochs, batch_size=4, learning_rate=3e-3,
                      betas=(0.9, 0.95), weight_decay=0.1,
                      lr_decay=True, warmup_tokens=tokens_per_epoch, final_tokens=train_epochs*tokens_per_epoch,
                      ckpt_path='brain_model_4l_4h_good.pt',
                      num_workers=0)
trainer = Trainer(model, train_dataset, test_dataset, tconf)

# train the model
#trainer.train()

# load the state of the best model we've seen based on early stopping
checkpoint = torch.load('brain_model_4l_4h_good.pt')
model.load_state_dict(checkpoint)


# to sample we also have to technically "train" a separate model for the first token in the sequence
# we are going to do so below simply by calculating and normalizing the histogram of the first token
counts = torch.ones(256)  # start counts as 1 not zero, this is called "smoothing"
rp = torch.randperm(len(train_dataset))
nest = 1000  # how many images to use for the estimation
for i in range(nest):
    a, _ = train_dataset[int(rp[i])]
    t = a[0].item()  # index of first token in the sequence
    counts[t] += 1
prob = counts / counts.sum()

seed = 5
set_seed(seed)
# create samples
n_samples = 4
start_pixel = np.random.choice(np.arange(256), size=(n_samples, 1), replace=True, p=prob)
start_pixel = torch.from_numpy(start_pixel).to(trainer.device)
pixels = sample(model, start_pixel, 32 * 32 - 1, temperature=1.0, sample=True, top_k=100)


set_seed(42)
# for visualization we have to invert the permutation used to produce the pixels
iperm = torch.argsort(train_dataset.perm)

# save samples
for i in range(n_samples):
    sample = pixels[i][iperm].cpu().view(32, 32).numpy().astype(np.uint8)
    cv2.imwrite(os.path.join(os.getcwd(), '..\\data\\samples', str(seed) + str(i + 1) + ".jpg"), sample)

