import torch
import torchvision
import torch.nn as nn
import random
import numpy as np
from mlp_mixer import MLPMixer
from patch_attacks import block_patchwise_grad_l2, patchwise_grad_l2

random.seed(2112)
np.random.seed(2112)
torch.manual_seed(2112)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

NUM_WORKERS = 2
BATCH_SIZE = 128
BASE_LOCATION = '/content/drive/MyDrive/bitirme/'
MODEL_NAME = "cifar10_adv_84"

test_transform = torchvision.transforms.Compose([ 
        # torchvision.transforms.Resize(size=(64,64)),      
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) # CIFAR
        # torchvision.transforms.Normalize((0.4802, 0.4481, 0.3975),(0.2770, 0.2691, 0.2821)) # TIM
        ])

test_data = torchvision.datasets.CIFAR10(root='./data', train=False, transform=test_transform, target_transform=None, download=True)
# test_data = TinyImageNet(root='./', train=False, transform=test_transform, target_transform=None, download=False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS )


checkpoint = torch.load(BASE_LOCATION + MODEL_NAME + "_checkpoint.pt", map_location='cuda:0')
last_epoch = checkpoint['epoch']
model = MLPMixer(*checkpoint['model_params']).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
try:
    IMG_DIMS, PATCH_SIZE, OUT_CLASSES, NUM_BLOCKS, HIDDEN_DIMS, TOKEN_DIMS, CHANNEL_DIMS, STOCHASTIC_DEPTH, DROPOUT = checkpoint['model_params']
except:
    IMG_DIMS, PATCH_SIZE, OUT_CLASSES, NUM_BLOCKS, HIDDEN_DIMS, TOKEN_DIMS, CHANNEL_DIMS, STOCHASTIC_DEPTH = checkpoint['model_params']
    DROPOUT = 0
print(f"""ImgDims:{IMG_DIMS} PatchSize:{PATCH_SIZE} OutClasses:{OUT_CLASSES} Blocks:{NUM_BLOCKS} HiddenDims:{HIDDEN_DIMS} TokenDims:{TOKEN_DIMS} ChannelDims:{CHANNEL_DIMS} StochasticDepth:{STOCHASTIC_DEPTH} Dropout:{DROPOUT}""")

# Calculate the L2 NORMS

model.eval()
patches = (IMG_DIMS[1]//PATCH_SIZE)*(IMG_DIMS[2]//PATCH_SIZE)
loss_func = nn.CrossEntropyLoss().to(device)
block_norms = []
img_norms = []
USE_RELU = False # Use ReLU when calculating L2 norm. 

for imgs, labels in test_loader:
    imgs = imgs.to(device)
    labels = labels.to(device)
    imgs.requires_grad = True
    outputs = model(imgs)
    
    block_patch_grad_norms = block_patchwise_grad_l2(imgs, labels, model, PATCH_SIZE, loss_func, device, relu=True)
    img_patch_grad_norms = patchwise_grad_l2(imgs, PATCH_SIZE, relu=USE_RELU)

    block_norms.append(block_patch_grad_norms)
    img_norms.append(img_patch_grad_norms)

img_norms = torch.cat(img_norms, dim=0).unsqueeze(1)
block_norms = torch.cat(block_norms, dim=0)
all_norms = torch.cat((img_norms.detach().cpu(), block_norms.detach().cpu()), dim=1)
relu = "_relu" if USE_RELU else ""
torch.save(all_norms, f"{BASE_LOCATION}{MODEL_NAME}{relu}_patch_norms.pt") # save the norms