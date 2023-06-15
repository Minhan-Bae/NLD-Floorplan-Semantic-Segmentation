# This script provides a custom implementation of the Stochastic Gradient Descent (SGD) optimizer with a NaN gradient handler.
# It extends the torch.optim.SGD optimizer and modifies the step() function to handle NaN gradients.
# The function returns a flag indicating the presence of NaN gradients and the loss, if available.

import torch

class SGD_NanHandler(torch.optim.SGD):
    def __init__(self, params, lr=0.1, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        super(SGD_NanHandler, self).__init__(params, lr, momentum, dampening, weight_decay, nesterov)

    @torch.no_grad()
    def step_handleNan(self, closure=None):
        """
        Performs a single optimization step, handling NaN gradients.

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss. Default is None.

        Returns:
            tuple: A tuple containing the following elements:
                - flag (bool): Indicates if NaN gradients were detected.
                - loss (float or None): The loss value if provided by the closure. Default is None.
        """
        loss = None
        flag = False
        
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Check for NaN gradients and return flag and loss if detected.
                if True in torch.isnan(p.grad):
                    flag = True
                    return flag, loss
                
                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                p.add_(d_p, alpha=-group['lr'])

        return flag, loss
