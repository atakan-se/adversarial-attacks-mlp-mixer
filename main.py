import torch
import torchvision
import torch.nn as nn
import random
import numpy as np
from augmentation import RandAug, apply_mixup
from mlp_mixer import MLPMixer
from utils import LinearDecay

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

BATCH_SIZE = 128
NUM_WORKERS = 2
EPOCHS = 300
VAL_INTERVAL = 10 # validate every n epochs
RAND_AUG_N = 2
RAND_AUG_M = 15
NUM_BLOCKS = 12
HIDDEN_DIMS = 256
LR = 0.001
WD = 0
TOKEN_DIMS = HIDDEN_DIMS//2
CHANNEL_DIMS = HIDDEN_DIMS*4
STOCHASTIC_DEPTH = 0.1
WARMUP_STEPS = 4000
LINEAR_DECAY_STEPS = 250000
MIXUP_ALPHA = 0.5

train_transform = torchvision.transforms.Compose([       
            torchvision.transforms.RandomHorizontalFlip(),
            RandAug(RAND_AUG_N, RAND_AUG_M),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
test_transform = torchvision.transforms.Compose([       
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

train_data = torchvision.datasets.CIFAR100(root='./data', train=True, transform=train_transform, target_transform=None, download=True)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS )
test_data = torchvision.datasets.CIFAR100(root='./data', train=False, transform=test_transform, target_transform=None, download=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS )

model = MLPMixer((3,32,32), 4, 100, NUM_BLOCKS, HIDDEN_DIMS, TOKEN_DIMS, CHANNEL_DIMS, STOCHASTIC_DEPTH).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
linear_decay = LinearDecay(optimizer, LR, 0 , LINEAR_DECAY_STEPS) # Linear decay will be applied after warmup
scheduler = LinearDecay(optimizer, 0, LR, WARMUP_STEPS, linear_decay) # Scheduler will act as warmup first, then use linear decay
loss_func = nn.CrossEntropyLoss().to(device)

best_accuracy = 0.0
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for imgs, labels in train_loader:
        # Get/Augment inputs and forward
        if MIXUP_ALPHA > 0:
            imgs, labels = apply_mixup(imgs, labels, MIXUP_ALPHA, OUT_CLASSES)
        outputs = model(imgs.to(device))
        # Calculate training loss
        loss = loss_func(outputs, labels.to(device))
        train_loss += loss.item()
        # Backprop
        scheduler.zero_grad()
        loss.backward()
        scheduler.step()
    
    print(f"Epoch:{epoch} Train loss:{train_loss:.2f}")    # Validation
    if epoch%VAL_INTERVAL: continue
    model.eval()
    with torch.no_grad():
        correct = 0
        for imgs, labels in train_loader:
            outputs = model(imgs.to(device))
            predictions = torch.max(outputs, dim=1)[1].cpu()
            correct += (predictions==labels).sum().item()
        print("Train acc.:", correct / len(train_data))
        # Test accuracy:
        correct = 0
        for imgs, labels in test_loader:
            outputs = model(imgs.to(device))
            predictions = torch.max(outputs, dim=1)[1].cpu()
            correct += (predictions==labels).sum().item()
        print("Validation acc.:", correct / len(test_data))
    
    if correct / len(test_data) > best_accuracy:
        best_accuracy = correct / len(test_data)
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_func }, 
                    'model_checkpoint.pt')
