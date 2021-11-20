import torch
from torchvision import datasets, transforms
import os
from PIL import Image

class TinyImageNet(datasets.VisionDataset):
    def __init__(self, root='./datasets', train=True, transform=None, target_transform=None, download=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.train = train
        self.root = root

        if download or 'tiny-imagenet-200' not in os.listdir(root):
            raise RuntimeError('You must download the dataset manually. You can use the tinyimagenet.sh file for this.')

        root = os.path.join(root, 'tiny-imagenet-200')
        with open(root + '/wnids.txt', 'r') as f:
            class_names = f.read().split('\n')[:-1]
            self.classes = dict( zip(class_names, [i for i in range(len(class_names))])) # Dictionary for indices of classes
        
        # #Try to get data from Pickled file
        # fname = 'tinyimagenet_train.pt' if self.train else 'tinyimagenet_val.pt'
        # if fname in os.listdir(self.root):
        #     print("Using the previously saved tensors for dataset.")
        #     data = torch.load(fname)
        #     self.imgs = data['imgs']
        #     self.labels = data['labels']
        #     return

        # Read images manually
        print("Creating dataset...")   
        imgs = []
        labels = []
        to_tensor = transforms.ToTensor()
        if self.train:
            for folder in os.listdir(root + '/train'):
                idx = self.classes[folder]
                folder_path = os.path.join(root, 'train', folder, 'images')
                for img_name in os.listdir(folder_path):
                    img = Image.open(folder_path + '/' + img_name).convert('RGB')
                    imgs.append(to_tensor(img))
                    labels.append(idx)
            self.imgs = torch.stack(imgs)
            self.labels = torch.tensor(labels, dtype=torch.long)
            #torch.save({'imgs':self.imgs, 'labels':self.labels}, self.root + '/tinyimagenet_train.pt')
        
        else:
            with open(root + '/val/val_annotations.txt', 'r') as f:
                val_classes = f.read().split('\n')[:-1]
                val_classes = [i.split('\t')[:2] for i in val_classes]
                val_classes = dict(val_classes) # class for each img file in val set

            folder_path = os.path.join(root, 'val', 'images')
            for img_name in os.listdir(folder_path):
                img = Image.open(folder_path + '/' + img_name).convert('RGB')
                imgs.append(to_tensor(img))
                idx = self.classes[val_classes[img_name]]
                labels.append(idx)
            self.imgs = torch.stack(imgs)
            self.labels = torch.tensor(labels, dtype=torch.long)
            #torch.save({'imgs':self.imgs, 'labels':self.labels}, self.root + '/tinyimagenet_val.pt')
                 
    def __getitem__(self, idx):
        img, label = self.imgs[idx], self.labels[idx]
        img =  transforms.ToPILImage()(img)

        if self.transform is not None:
            img = self.transform(img)
        
        if self.target_transform is not None:
            label = self.target_transform(label)
        
        return img, label

    def __len__(self):
        return len(self.labels)