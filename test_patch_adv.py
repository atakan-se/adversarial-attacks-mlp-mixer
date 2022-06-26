import torch
import torchvision
import torch.nn as nn
import random
import numpy as np
from mlp_mixer import MLPMixer
from patch_attacks import patch_FGSM

random.seed(2112)
np.random.seed(2112)
torch.manual_seed(2112)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

NUM_WORKERS = 2
BATCH_SIZE = 128
BASE_LOCATION = '/content/drive/MyDrive/bitirme/'
MODEL_NAME = "cifar10_adv_84"
PATCHES = 5 # NUMBER OF PATCHES TO ATTACK
EPSILON = 1 # ATTACK STRENGTH

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

all_norms = torch.load(f"{BASE_LOCATION}{MODEL_NAME}_patch_norms.pt")

loss_func = nn.CrossEntropyLoss().to(device)
correct_adv = torch.zeros(NUM_BLOCKS+1)
correct_original = 0
total = 0
model.eval()
sorted_norms, sorted_idx = torch.sort(all_norms, dim=2, descending=True)
for imgs, labels in test_loader:
    block_adv_imgs = patch_FGSM(imgs, labels, model, loss_func, EPSILON, sorted_idx[i*BATCH_SIZE:(i+1)*BATCH_SIZE,:,:PATCHES], PATCH_SIZE)
    with torch.no_grad():
        for bidx in range(NUM_BLOCKS+1):
            outputs = model(block_adv_imgs[bidx].to(device))
            predictions = torch.max(outputs, dim=1)[1].cpu()
            correct_adv[bidx] += (predictions==labels).sum().item()
        outputs = model(imgs.to(device))
        predictions = torch.max(outputs, dim=1)[1].cpu()
        correct_original += (predictions==labels).sum().item()
        total += predictions.shape[0]

print("Accuracy for each block:", correct_adv/total)
print("Clean accuracy:", correct_original/total)