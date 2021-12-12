import torch
import torchvision
import torch.nn as nn
import random
import numpy as np
from augmentation import RandAug, apply_mixup
from mlp_mixer import MLPMixer
from utils import LinearDecay

random.seed(2112)
np.random.seed(2112)
torch.manual_seed(2112)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

PATCH_SIZE = 4
BATCH_SIZE = 128
NUM_WORKERS = 2
EPOCHS = 300
VAL_INTERVAL = 10 # validate every n epochs
RAND_AUG_N = 2
RAND_AUG_M = 15
NUM_BLOCKS = 12
HIDDEN_DIMS = 256
LR = 0.001
WD = 4e-5
TOKEN_DIMS = HIDDEN_DIMS//2
CHANNEL_DIMS = HIDDEN_DIMS*4
STOCHASTIC_DEPTH = 0.1
DROPOUT = 0
WARMUP_EPOCHS = 5
SCHEDULER_EPOCHS = EPOCHS - WARMUP_EPOCHS
MIXUP_ALPHA = 0.5

train_transform = torchvision.transforms.Compose([       
            torchvision.transforms.RandomHorizontalFlip(),
            #torchvision.transforms.RandomResizedCrop(size=(64,64)),
            RandAug(RAND_AUG_N, RAND_AUG_M),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
test_transform = torchvision.transforms.Compose([ 
        #torchvision.transforms.Resize(size=(64,64)),      
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

train_data = torchvision.datasets.CIFAR100(root='./data', train=True, transform=train_transform, target_transform=None, download=True)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS )
test_data = torchvision.datasets.CIFAR100(root='./data', train=False, transform=test_transform, target_transform=None, download=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS )

IMG_DIMS = train_data[0][0].shape
OUT_CLASSES = 100 # TODO: change to dynamic
SCHEDULER_STEPS = SCHEDULER_EPOCHS * len(train_data)
WARMUP_STEPS = WARMUP_EPOCHS * len(train_data)

model = MLPMixer(IMG_DIMS, PATCH_SIZE, OUT_CLASSES, NUM_BLOCKS, HIDDEN_DIMS, TOKEN_DIMS, CHANNEL_DIMS, STOCHASTIC_DEPTH, DROPOUT).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
scheduler_after_warmup = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, SCHEDULER_STEPS)
scheduler = LinearDecay(optimizer, 0, LR, WARMUP_STEPS, scheduler_after_warmup) # Scheduler will act as warmup first, then use linear decay
scheduler.step() # start warmup
loss_func = nn.CrossEntropyLoss().to(device)

print(f"""ImgDims:{IMG_DIMS} PatchSize:{PATCH_SIZE} OutClasses:{OUT_CLASSES} Blocks:{NUM_BLOCKS} HiddenDims:{HIDDEN_DIMS} TokenDims:{TOKEN_DIMS} ChannelDims:{CHANNEL_DIMS} StochasticDepth:{STOCHASTIC_DEPTH}
Dropout:{DROPOUT} Optimizer:{optimizer} LR:{LR}, WD:{WD}, WarmupSteps:{WARMUP_STEPS} SchedulerSteps:{SCHEDULER_STEPS}
BatchSize:{BATCH_SIZE} Epochs:{EPOCHS} MixupAlpha:{MIXUP_ALPHA} RandAugN:{RAND_AUG_N} RandAugM:{RAND_AUG_M} """)

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
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
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
                    'val_acc': best_accuracy,
                    'loss': train_loss,
                    'model_params': (IMG_DIMS, PATCH_SIZE, OUT_CLASSES, NUM_BLOCKS, HIDDEN_DIMS, TOKEN_DIMS, CHANNEL_DIMS, STOCHASTIC_DEPTH)}, 
                    'model_checkpoint.pt')
