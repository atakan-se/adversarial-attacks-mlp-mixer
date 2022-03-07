import torch
import numpy as np

def FGSM(imgs, labels, model, loss_func, eps, clamp=(0,1)):
    """
    Applies Fast Gradient Sign Method introduced in "Explaining and harnessing adversarial examples" 
    by Ian J. Goodfellow, Jonathon Shlens, Christian Szegedy (https://arxiv.org/abs/1412.6572)
    """
    model.zero_grad()
    device = next(model.parameters()).device
    adv_imgs = imgs.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    adv_imgs.requires_grad = True
    preds = model(adv_imgs)
    loss = loss_func(preds, labels)
    loss.backward()
    adv_imgs = adv_imgs + eps * adv_imgs.grad.sign()

    # return torch.clamp(adv_imgs.detach(), *clamp) # clamp between 0 and 1
    return adv_imgs.detach()

def PGD(imgs, labels, model, loss_func, eps, steps, step_size, clamp=(0,1), 
        targeted=False,  # With targeted mode, labels are target classes
        random_init=True): # Random init for delta, meaning images start with some random perturbation
    """
    Projected Gradient Descent method based on Towards Deep Learning Models Resistant to Adversarial Attacks
    by Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, Adrian Vladu (https://arxiv.org/pdf/1706.06083.pdf)
    Uses L_inf norm for all projections
    """
    model.zero_grad()
    device = next(model.parameters()).device
    adv_imgs = imgs.clone().detach().to(device)
    labels = labels.clone().detach().to(device)

    if random_init: # initialization for delta (otherwise delta=0)
        adv_imgs = adv_imgs + torch.empty_like(imgs).uniform_(-eps, eps).to(device)
        adv_imgs = torch.clamp(adv_imgs, *clamp)

    for _ in range(steps):
        adv_imgs.requires_grad = True
        preds = model(adv_imgs)
        loss = loss_func(preds, labels)
        loss.backward()
        with torch.no_grad():
            if targeted:
                adv_imgs = adv_imgs - adv_imgs.grad.sign() * step_size
            else:
                adv_imgs = adv_imgs + adv_imgs.grad.sign() * step_size

            adv_imgs = torch.max(torch.min(adv_imgs, imgs.to(device) + eps), imgs.to(device) -eps) # Move back to Ln-Ball around original image
            # adv_imgs = torch.clamp(adv_imgs, *clamp) # Clamp to natural iamge interval
      
    return adv_imgs.detach()

def DeepFool(imgs, labels, model, loss_func, max_steps):
    """
    DeepFool method based on DeepFool: a simple and accurate method to fool deep neural networks
    by Seyed-Mohsen Moosavi-Dezfooli, Alhussein Fawzi, Pascal Frossard (https://arxiv.org/abs/1511.04599)
    """
    model.zero_grad()
    device = next(model.parameters()).device
    adv_imgs = imgs.clone().detach()
    labels = labels.clone().detach().to(device)

    for idx in range(len(adv_imgs)): # for each image in a batch
        sample = adv_imgs[idx].clone().detach().to(device)
        sample.requires_grad = True
        step = 0 
        label = labels[idx].item()
        output = model(sample.unsqueeze(0))
        if output.data.shape!=1: output = output.reshape(1,-1)
        _, pred = torch.max(output.cpu().data, dim=1)
        while pred.item() == label and step < max_steps:
            min_perturb = np.inf # dummy value
            w_l = None
            fs = output[0] - output[0, label]
            output[0, 0].backward(retain_graph=True)
            w_0 = sample.grad.data.cpu().numpy().copy()
            sample.grad.zero_()
            for k in range(output.shape[1]): # for all classes
                if k==label:continue # skip original label
                output[0, k].backward(retain_graph=True)
                w_k = sample.grad.data.cpu().numpy().copy() - w_0
                sample.grad.zero_()
                perturb = abs(fs[k]) / np.linalg.norm(w_k)
                if perturb < min_perturb:
                    min_perturb = perturb
                    w_l = w_k.copy()
            r_i = (perturb.detach().cpu() + 1e-6) * w_l / np.linalg.norm(w_l)
            adv_imgs[idx] += r_i 
            step += 1
    
    return adv_imgs.detach()
