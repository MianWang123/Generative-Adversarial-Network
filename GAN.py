"""
CIS522-Deep Learning for Data Science: Generative Adversarial Network
Author: Mian Wang  
Time: 3/6/20
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision as tv
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from glob import glob
from PIL import Image
from torch.utils.data.sampler import SubsetRandomSampler
from torchsummary import summary

# Step 0: Set GPU in google colab and launch tensorboard
device = torch.device('cuda:0' if torch.cuda.is_available() else 'gpu')
%load_ext tensorboard


# Step 1: Import the dataset
UT_transforms = transforms.Compose([ transforms.Resize((64, 64)),
                                      transforms.ToTensor()      ])
UT_dataset = dset.ImageFolder(root='DATASETS/UTZappos50K', transform=UT_transforms)


# Step 2: Split the data into train, validation, and test set(7:2:1), then set batch size to 64
idx = list(range(len(UT_dataset)))
np.random.shuffle(idx)

split1, split2 = int(np.floor(len(idx)*0.7)), int(np.floor(len(idx)*0.9))
ut_train_sampler = SubsetRandomSampler(idx[:split1])
ut_val_sampler = SubsetRandomSampler(idx[split1:split2])
ut_test_sampler = SubsetRandomSampler(idx[split2:])

batch_size = 64
ut_train_loader = DataLoader(UT_dataset, batch_size=batch_size, sampler=ut_train_sampler)
ut_val_loader = DataLoader(UT_dataset, batch_size=batch_size, sampler=ut_val_sampler)
ut_test_loader = DataLoader(UT_dataset, batch_size=batch_size, sampler=ut_test_sampler)


# Step 3: Create a normally distributed latent vector z
feature_length = 128
z = torch.randn(size=[batch_size,feature_length,1,1], device=device)


# Step 4: Establish the generator, use summary to check the network
class Generator(nn.Module):
  def __init__(self):
    super(Generator,self).__init__()
    self.l1 = nn.ConvTranspose2d(in_channels=feature_length, out_channels=512, kernel_size=4, stride=1, padding=0)
    self.bn1 = nn.BatchNorm2d(512)  
    self.l2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1)
    self.bn2 = nn.BatchNorm2d(256)  
    self.l3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)
    self.bn3 = nn.BatchNorm2d(128)  
    self.l4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
    self.bn4 = nn.BatchNorm2d(64)  
    self.l5 = nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=4, stride=2, padding=1)
  
  def forward(self, z):
    z = F.leaky_relu(self.bn1(self.l1(z)))
    z = F.leaky_relu(self.bn2(self.l2(z)))
    z = F.leaky_relu(self.bn3(self.l3(z)))
    z = F.leaky_relu(self.bn4(self.l4(z)))
    img = torch.tanh(self.l5(z))
    return img

generator = Generator().to(device)
summary(generator, (feature_length,1,1))


# Step 5: Build the discriminator
class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator,self).__init__()
    self.l1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1)
    self.bn1 = nn.BatchNorm2d(64)
    self.l2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
    self.bn2 = nn.BatchNorm2d(128)
    self.l3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)
    self.bn3 = nn.BatchNorm2d(256)
    self.l4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1)
    self.bn4 = nn.BatchNorm2d(512)
    self.l5 = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=0)
    
  def forward(self, img):
    img = F.relu(self.bn1(self.l1(img)))
    img = F.relu(self.bn2(self.l2(img)))
    img = F.relu(self.bn3(self.l3(img)))
    img = F.relu(self.bn4(self.l4(img)))
    y = torch.sigmoid(self.l5(img))
    return y.view(-1)

discriminator = Discriminator().to(device)
summary(discriminator, (3,64,64))


# Step 6: Set the generator loss function
def generator_loss(generator, discriminator, z):
  criterion = nn.BCELoss()
  ones = torch.ones(z.size(0), device=device)   
  loss_gen = criterion(discriminator(generator(z)), ones)
  return loss_gen
 
 
# Step 7: Set the discriminator loss function
def discriminator_loss(discriminator, generator, real_img, z, fuzzy=False):
  criterion = nn.BCELoss()
  if fuzzy:
    ones = torch.ones(real_img.size(0), device=device)*0.9
    zeros = torch.zeros(z.size(0), device=device)
  else:
    ones = torch.ones(real_img.size(0), device=device)
    zeros = torch.zeros(z.size(0), device=device)

  fake_img = generator(z).detach()
  loss_real = criterion(discriminator(real_img), ones)
  loss_fake = criterion(discriminator(fake_img), zeros)   
  return (loss_real+loss_fake)/2


# Step 8: Set the optimizer for generator and discriminator
optimizer_gen = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_dis = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))


# Step 9: Displays the images in a (rows * cols) grid
def generate_images(generator, z, rows, cols, name=None, show=True):
  imgs = generator(z)
  imgs = imgs[:rows*cols].cpu().detach()
  fig, ax = plt.subplots(rows, cols)
  count = 0
  imgs = np.reshape(imgs, (imgs.shape[0],3,64,64))
  imgs = np.clip(imgs, a_min=0, a_max=255) 

  for row in range(rows):
    for col in range(cols):
      img = imgs[count]
      ax[row,col].imshow(img.permute(1,2,0))
      ax[row,col].axis('off')
      count += 1

  fig.suptitle(name)
  fig.savefig("%s.png"%(name))
  if(show): plt.show()
  else: plt.close()

    
# Step 10: Train the GAN model
logger = SummaryWriter('logs/GAN')
epochs = 10
step = 0

for epoch in range(epochs):
  for i, (x,_) in enumerate(ut_train_loader):
    step += 1
    real_img = x.to(device)
    # choose a normally distributed latent vector
    z = torch.randn(size=[x.size(0),feature_length,1,1], device=device)

    # train generator
    optimizer_gen.zero_grad()
    loss_gen = generator_loss(generator, discriminator, z)
    loss_gen.backward()
    optimizer_gen.step()

    # train discriminator
    optimizer_dis.zero_grad()
    loss_dis = discriminator_loss(discriminator, generator, real_img, z, fuzzy=False)
    loss_dis.backward()
    optimizer_dis.step()

    # add scalar to tensorboard
    logger.add_scalar('Generator Loss', loss_gen, step)
    logger.add_scalar('Discriminator Loss', loss_dis, step)

    if step % 100 == 0: 
      print('[%d/%d : %d] Generator Loss: %.4f\t Discriminator Loss: %.4f\t'%(epoch+1, epochs, step, loss_gen, loss_dis))
    if step % 500 == 0:
      generate_images(generator, z, 2, 5, name='generator image', show=True)
   
  
   # Step 11: Display the training loss in tensorboard
   %tensorboard --logdir 'logs/GAN'
