import torch

def fgsm_training(model, loss_func, optimizer, imgs, labels, step_size, eps, scheduler=None, device='cuda'):
    """
    Fast adversarial training from FAST IS BETTER THAN FREE: REVISITING ADVERSARIAL TRAINING
    by Eric Wong, Leslie Rice, J. Zico Kolter

    imgs and labels should be a single batch.
    Returns training loss
    """
    delta = torch.zeros_like(imgs).to(device)
    delta.uniform_(-eps, +eps) # intialize randomly
    delta.requires_grad = True
    output1 = model(imgs.to(device) + delta)
    loss1 = loss_func(output1, labels.to(device))
    loss1.backward()
    delta.data = torch.clamp(delta + step_size * torch.sign(delta.grad), -eps, eps) # step delta (FGSM)
    # Train model:
    output2 = model(imgs.to(device) + delta)
    # Calculate training loss
    loss = loss_func(output2, labels.to(device))
    # Backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if scheduler is not None: scheduler.step()
    return loss.item()