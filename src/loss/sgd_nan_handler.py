import torch

class SGD_NanHandler(torch.optim.SGD):
    def __init__(self, params, lr=0.1, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        # SGD_NanHandler 클래스의 생성자입니다.
        # params: 최적화할 파라미터들
        # lr: 학습률 (learning rate), 기본값은 0.1
        # momentum: 모멘텀, 기본값은 0
        # dampening: 감쇠 계수, 기본값은 0
        # weight_decay: 가중치 감쇠 계수, 기본값은 0
        # nesterov: Nesterov 모멘텀 사용 여부, 기본값은 False
        super(SGD_NanHandler, self).__init__(params, lr, momentum, dampening, weight_decay, nesterov)

    @torch.no_grad()
    def step_handleNan(self, closure=None):
        """
        NaN 그래디언트를 처리하면서 단일 최적화 단계를 수행합니다.

        Args:
            closure (callable, optional): 모델을 재평가하고 손실을 반환하는 클로저입니다. 기본값은 None.

        Returns:
            tuple: flag와 loss 값을 포함하는 튜플입니다.
                - flag (bool): NaN 그래디언트가 검출되었는지 여부를 나타냅니다.
                - loss (float 또는 None): 클로저에서 제공된 손실 값입니다. 기본값은 None입니다.
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

            for param in group['params']:
                if param.grad is None:
                    continue
                
                # NaN 그래디언트를 확인하고 검출되면 플래그와 손실을 반환합니다.
                if True in torch.isnan(param.grad):
                    flag = True
                    return flag, loss
                
                d_p = param.grad
                if weight_decay != 0:
                    d_p = d_p.add(param, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[param]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                param.add_(d_p, alpha=-group['lr'])

        return flag, loss
