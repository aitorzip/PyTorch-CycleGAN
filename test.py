import argparse
import os
import sys

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image

from datasets import ImageDataset
from models import Generator

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1,
                    help='size of the batches')
parser.add_argument('--dataroot', type=str, default='datasets/horse2zebra/',
                    help='root directory of the dataset')
parser.add_argument('--input_nc', type=int, default=1,
                    help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=1,
                    help='number of channels of output data')
parser.add_argument('--size', type=int, default=256,
                    help='size of the data (squared assumed)')
parser.add_argument('--cuda', action='store_true',
                    help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8,
                    help='number of cpu threads to use during batch generation')
parser.add_argument('--generator_A2B', type=str, default='output/netG_A2B.pth',
                    help='A2B generator checkpoint file')
parser.add_argument('--generator_B2A', type=str, default='output/netG_B2A.pth',
                    help='B2A generator checkpoint file')
opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# Networks
netG_A2B = Generator(opt.input_nc, opt.output_nc)
netG_B2A = Generator(opt.output_nc, opt.input_nc)

if opt.cuda:
    netG_A2B.cuda()
    netG_B2A.cuda()

# Load state dicts
netG_A2B.load_state_dict(torch.load(opt.generator_A2B))
netG_B2A.load_state_dict(torch.load(opt.generator_B2A))

# Set model's test mode
netG_A2B.eval()
netG_B2A.eval()

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)

# Dataset loader
transforms_ = [
    transforms.ToTensor(),
    transforms.Normalize(tuple([0.5] * opt.output_nc), tuple([0.5] * opt.output_nc))
]

dataset = ImageDataset(opt.dataroot, transforms_=transforms_, mode='test', grayscale=(opt.input_nc == 1))
dataloader = DataLoader(dataset, batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)

# Create output dirs if they don't exist
if not os.path.exists('output/A'):
    os.makedirs('output/A')
if not os.path.exists('output/B'):
    os.makedirs('output/B')

for i, batch in enumerate(dataloader):
    # Set model input
    real_A = Variable(input_A.copy_(batch['A']))
    real_B = Variable(input_B.copy_(batch['B']))

    # Generate output
    fake_B = 0.5 * (netG_A2B(real_A).data + 1.0)
    fake_A = 0.5 * (netG_B2A(real_B).data + 1.0)

    # Save image files
    save_image(fake_A, f'output/A/{(i + 1):04d}.png')
    save_image(fake_B, f'output/B/{(i + 1):04d}.png')

    sys.stdout.write(f'\rGenerated images {(i + 1):04d} of {len(dataloader):04d}')

sys.stdout.write('\n')
