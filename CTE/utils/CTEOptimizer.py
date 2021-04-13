import torch
import numpy as np

from torch.optim import optimizer


class CTEOptimizer(optimizer.Optimizer):
    """
    The implementation here takes the square root of the gradient average before
    adding epsilon (note that TensorFlow interchanges these two operations). The effective
    learning rate is thus :math:`\alpha/(\sqrt{v} + \epsilon)` where :math:`\alpha`
    is the scheduled learning rate and :math:`v` is the weighted moving average
    of the squared gradient.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        opt_to_use(array): list of the optimizer to use, size of the number of elements in params
            0 - use only SGD, 1 - use RMSprop
        lr (float/array, optional): learning rate (default: 1e-1)
        momentum (float, optional): momentum factor (default: 0.0)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-9)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0.1)

    """

    def __init__(self, params, lr=1e-1, alpha=0.99, eps=1e-9, weight_decay=0.1, momentum=1, RMS_support = False, batch_size = 100):
        if type(lr) is np.ndarray:
            if np.any(lr < 0.0):
                raise ValueError("Invalid learning rate: at least one of the values is negative")
        else:
            if not 0.0 <= lr:
                raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))
        if not 0.0 <= batch_size:
            raise ValueError("Invalid alpha value: {}".format(alpha))

        defaults = dict(lr=lr, alpha=alpha, momentum=momentum, eps=eps, weight_decay=weight_decay, RMS_support=RMS_support, batch_size=batch_size)
        super(CTEOptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(CTEOptimizer, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('momentum', 0)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                RMS_support = group['RMS_support']

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    if RMS_support:
                        state['square_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    else:
                        state['square_avg'] = torch.ones_like(p.data, memory_format=torch.preserve_format)
                    if group['momentum'] > 0:
                        state['momentum_buffer'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)


                square_avg = state['square_avg']
                alpha = group['alpha']

                state['step'] += 1

                if RMS_support:
                    square_avg = alpha*square_avg + (1-alpha)*(grad.pow(2))
                    grad = (grad*group['batch_size'])/(torch.sqrt(square_avg.add_(group['eps'])))
                    # square_avg.mul_(alpha).addcmul_(1 - alpha, grad, grad)

                current_LR = group['lr']
                current_WD = group['weight_decay']


                buf = state['momentum_buffer']
                buf = alpha * buf - current_LR * current_WD * p.data - (current_LR/group['batch_size']) * grad
                p.data.add_(buf)

                # if group['weight_decay'] != 0:
                #     grad = grad.add(group['weight_decay'], p.data)
                #
                # avg = square_avg.sqrt().add_(group['eps'])
                #
                # if group['momentum'] > 0:
                #     buf = state['momentum_buffer']
                #     buf.mul_(group['momentum']).addcdiv_(grad, avg)
                #     p.data.add_(-group['lr'], buf)
                # else:
                #     p.data.addcdiv_(-group['lr'], grad, avg)
        return loss
