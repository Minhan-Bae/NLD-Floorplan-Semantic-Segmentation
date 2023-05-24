import torch.optim as optim

def get_optimizer(model, args):
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.base_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    return optimizer
