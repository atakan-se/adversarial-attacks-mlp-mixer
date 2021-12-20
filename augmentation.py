from PIL import Image, ImageOps, ImageEnhance
import random
import numpy as np
from torch.nn.functional import one_hot
import torch 

# Image transformation functions. All of them expect a PIL image.
def identity(img, _=0):
    return img

def rotate(img, theta):
    return img.rotate(theta)

def posterize(img, val):
    return ImageOps.posterize(img, int(val))

def sharpness(img, val):
    return ImageEnhance.Sharpness(img).enhance(val)

def translateX(img, offset):
    return img.transform(img.size, Image.AFFINE, (1, 0, offset, 0, 1, 0))

def translateY(img, offset):
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, offset))

def auto_contrast(img, _=0):
    return ImageOps.autocontrast(img)

def solarize(img, val):
    return ImageOps.solarize(img, val)

def contrast(img, val):
    return ImageEnhance.Contrast(img).enhance(val)

def shearX(img, offset):
    return img.transform(img.size, Image.AFFINE, (1, offset, 0, 0, 1, 0))

def shearY(img, offset):
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, offset, 1, 0))

def equalize(img, _=0):
    return ImageOps.equalize(img)

def color(img, val):
    return ImageEnhance.Color(img).enhance(val)

def brightness(img, val):
    return ImageEnhance.Brightness(img).enhance(val)


# Implementation of RandAug
# Paper "RandAugment: Practical automated data augmentation with a reduced search space"
# by Ekin D. Cubuk, Barret Zoph, Jonathon Shlens, Quoc V. Le , Google Research, Brain Team
# at https://arxiv.org/abs/1909.13719
class RandAug():
    def __init__(self, N, M):
        assert 0 <= M <= 30 # Range for Magnitude is taken from the paper
        self.N = N # Number of sequential transforms
        self.M = M # Magnitude 
        # Define Policies for each function same as in the paper (min-max values)
        self.policy = {'identity':(identity, 0, 0),     
                       'rotate':(rotate, 0, 30),
                       'posterize':(posterize, 4, 8),
                       'sharpness':(sharpness, 0.1, 1.9),
                       'translateX':(translateX, 0, 0.33),
                       'translateY':(translateY, 0, 0.33),
                       'auto_contrast':(auto_contrast, 0, 0),
                       'solarize':(solarize, 0, 110),
                       'contrast':(contrast, 0.1, 1.9),
                       'shearX':(shearX, 0, 0.3),
                       'shearY':(shearY, 0, 0.3),
                       'equalize':(equalize, 0, 0),
                       'color':(color, 0.1, 1.9),
                       'brightness':(brightness, 0.1, 1.9),}
        
    def __call__(self, img):
        ops = random.sample(self.policy.keys(), self.N)
        out = img.copy()
        for op in ops:
            func, min_val, max_val = self.policy[op]
            val = (self.M/30) * (max_val - min_val) + min_val # Linear magnitude 
            if op in {'rotate', 'translateX', 'translateY', 'shearX', 'shearY' }:
                if random.random() < 0.5:
                    val *= -1
            out = func(out, val) # Call the function with argument
        return out


def apply_mixup(imgs, labels, alpha, num_classes):
    '''
    Mixup method from "Mmixup: Beyond Empirical Risk Minimization"
    by Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz
    at https://openreview.net/forum?id=r1Ddp1-Rb
    '''
    assert alpha>0
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(len(imgs))

    mixed_imgs = lam * imgs + (1 - lam) * imgs[idx, :]
    label_probs = lam * one_hot(labels, num_classes) + (1 - lam) * one_hot(labels[idx], num_classes)
    
    return mixed_imgs, label_probs

class Cutout():
    def __init__(self, mask_size):
        if isinstance(mask_size, int):
            self.mask_size = (mask_size, mask_size)
        else:
            self.mask_size = mask_size
    
    def __call__(self, img):
        img = np.asarray(img).copy()
        W, H = img.shape[:2]
        x = np.random.randint(0, W+1)
        y = np.random.randint(0, H+1)
        y_left = y - self.mask_size[0]//2
        y_left = 0 if y_left<0 else y_left
        y_right = y + self.mask_size[0]//2
        y_right = H if y_right > H else y_right
        x_left = x - self.mask_size[1]//2
        x_left = 0 if x_left<0 else x_left
        x_right = x + self.mask_size[1]//2
        x_right = H if x_right > W else x_right
        img[y_left:y_right, x_left:x_right] = (0,0,0)
        return Image.fromarray(img)
