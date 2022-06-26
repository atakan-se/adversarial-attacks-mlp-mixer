import torch
from utils import patchify

def patch_FGSM(imgs, labels, model, loss_func, eps, patches, patch_size, clamp=(0,1)):
    """ 
    Applies Fast Gradient Sign Method to the patches given.
    Patches has shape (N, Bl, P) where N is the batch, Bl is the block choosing the patch, and P is the chosen patches
    This way, same attack can be applied to multiple choosing criteria
    """
    model.zero_grad()
    if len(patches.shape)==2: patches = patches.unsqueeze(2)
    patch_count = patches.shape[-1]
    device = next(model.parameters()).device
    adv_imgs = imgs.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    block_adv_imgs = torch.cat(patches.shape[-2] * [imgs.clone().detach().to(device).unsqueeze(0)], dim=0)
    adv_imgs.requires_grad = True
    preds = model(adv_imgs)
    loss = loss_func(preds, labels)
    loss.backward()
    for block_idx in  range(block_adv_imgs.shape[0]):
        for img_idx in range(block_adv_imgs.shape[1]):
            for patch in range(patch_count):
                pid = patches[img_idx, block_idx, patch]
                py, px = pid//(imgs.shape[-2] // patch_size), pid%(imgs.shape[-1] // patch_size)
                block_adv_imgs[block_idx, img_idx, :, py*patch_size:(py+1)*patch_size,px*patch_size:(px+1)*patch_size] += eps * adv_imgs.grad[img_idx, :, py*patch_size:(py+1)*patch_size,px*patch_size:(px+1)*patch_size].sign()
    
    # return torch.clamp(adv_imgs.detach(), *clamp) # clamp between 0 and 1
    return block_adv_imgs

def patchwise_grad_l2(imgs, patch_size, relu=False):
    """
    Calculates the Patchwise L2 Norms for the Image gradient
    """
    if relu: grad = torch.nn.ReLU()(imgs.grad)
    else: grad = imgs.grad
    grad_patches = patchify(grad, patch_size, (patch_size,patch_size))
    patch_norms = torch.linalg.norm(grad_patches, dim=(4,5))
    patch_norms = torch.mean(patch_norms, dim=1).flatten(start_dim=1)
    return patch_norms


def block_patchwise_grad_l2(imgs, labels, model, patch_size, loss_func, device, relu=False):
    """
    Used for saving the backprop gradients through the MLP-Mixer in this project.
    Saves the L2 norm over the channel dimension and returns it.
    """
    batch_size = imgs.shape[0]
    patch_grad_norms = torch.zeros((batch_size, len(model.mixers), model.num_patches))
    def save_grad(i):
        def hook(grad):
            if relu: grad = torch.nn.ReLU()(grad)
            grad_norm = torch.norm(grad, dim=2, p=2)
            patch_grad_norms[:,i] = grad_norm
        return hook
    # MIXER 
    y = model.stem(imgs) # y is now N C H W
    y = torch.flatten(y, start_dim=2, end_dim=3) # y is now N C P
    y = torch.transpose(y, 1, 2) # y is now N P C
    
    for i, mixer_block in enumerate(model.mixers):
        y = mixer_block(y)
        y.register_hook(save_grad(i))
           
    y = model.layernorm(y)
    y = torch.mean(y, dim=1, keepdim=False)
    y = model.classifier(y)
    loss = loss_func(y, labels)
    loss.backward()

    return patch_grad_norms
