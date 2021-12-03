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
                self._optimizer.step()
        else:
            self._update_lr()
            self._optimizer.step()

    def _update_lr(self):
        self.lr += self.rate
        self.steps += 1
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = self.lr
    
    def zero_grad(self):
        self._optimizer.zero_grad()