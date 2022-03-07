import torch
import torchvision
import torch.nn as nn
import random
import numpy as np
from mlp_mixer import MLPMixer
from adversarial_attacks import FGSM, PGD, DeepFool
random.seed(2112)
np.random.seed(2112)
torch.manual_seed(2112)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

NUM_WORKERS = 2
BATCH_SIZE = 128
MODEL_PATH = '/content/drive/MyDrive/bitirme/cifar100_65_checkpoint.pt'
GPU_ID = 0
device = device if GPU_ID==None else device + ':' + str(GPU_ID)
ADV_ATTACK = FGSM # PGD #DeepFool # FGSM
if ADV_ATTACK==FGSM:
    params = {'eps':0.05} # FGSM
elif ADV_ATTACK==DeepFool:
    params = {'max_steps':2} #  DeepFool
else:
    params = {'eps':0.25, 'steps':7, 'step_size':2} # PGD

DATASET = 'cifar100'

test_transform = torchvision.transforms.Compose([ 
        torchvision.transforms.Resize(size=(64,64)),      
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

if DATASET=='cifar100':
    test_data = torchvision.datasets.CIFAR100(root='./data', train=False, transform=test_transform, target_transform=None, download=True)
else:
    test_data = torchvision.datasets.CIFAR10(root='./data', train=False, transform=test_transform, target_transform=None, download=True)

test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS )

checkpoint = torch.load(MODEL_PATH, map_location=device)
last_epoch = checkpoint['epoch']
model = MLPMixer(*checkpoint['model_params']).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
IMG_DIMS, PATCH_SIZE, OUT_CLASSES, NUM_BLOCKS, HIDDEN_DIMS, TOKEN_DIMS, CHANNEL_DIMS, STOCHASTIC_DEPTH, DROPOUT = checkpoint['model_params']

print(f"""ImgDims:{IMG_DIMS} PatchSize:{PATCH_SIZE} OutClasses:{OUT_CLASSES} Blocks:{NUM_BLOCKS} HiddenDims:{HIDDEN_DIMS} TokenDims:{TOKEN_DIMS} ChannelDims:{CHANNEL_DIMS} StochasticDepth:{STOCHASTIC_DEPTH} Dropout:{DROPOUT}""")


loss_func = nn.CrossEntropyLoss().to(device)
correct_adv = 0
correct_original = 0
total = 0
model.eval()

for imgs, labels in test_loader:
    adv_imgs = ADV_ATTACK(imgs, labels, model, loss_func, **params)
    with torch.no_grad():
        outputs = model(adv_imgs.to(device))
        predictions = torch.max(outputs, dim=1)[1].cpu()
        correct_adv += (predictions==labels).sum().item()
        outputs = model(imgs.to(device))
        predictions = torch.max(outputs, dim=1)[1].cpu()
        correct_original += (predictions==labels).sum().item()
        total += predictions.shape[0]

print(f"Test Acc:{correct_original / total}", f"Adv. Attack ({ADV_ATTACK.__name__}) Acc:{correct_adv / total }", f"Sample Size:{total}", sep='\n')