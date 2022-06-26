class LinearDecay():
    """ Applies linear change to the learning rate. After the warmup, uses a defined Scheduler. Can be used as Linear Warmup """

    def __init__(self, optimizer, lr_start, lr_end, n_steps, next_scheduler=None):
        self._optimizer = optimizer
        self.lr = lr_start
        self.n_steps = n_steps
        self.rate = (lr_end - lr_start) / n_steps
        self.next_scheduler = next_scheduler
        self.steps = 0
    
    def step(self):
        if self.n_steps==self.steps: # Done with the warmup
            if self.next_scheduler:
                self.next_scheduler.step()
        else:
            self._update_lr()

    def _update_lr(self):
        self.lr += self.rate
        self.steps += 1
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = self.lr

def patchify(I, patch_size, stride=[1,1]):
    """ 
    Takes Image I with (*, H, W); splits into patches and returns (*, PY, PX, PH, PW) 
    E.g.:  (3,32,32) image is split into patches of size 4 with stride 4: (3, 8, 8, 4, 4)
    """
    H,W = I.shape[-2:]
    ph,pw = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
    sh,sw = (stride, stride) if isinstance(stride, int) else stride
    assert not (((H - ph) % sh) or ((W - pw) % sw))
    return I.unfold(-2, ph, sh).unfold(-2, pw, sw)
