def adjust_learning_rate(args, optimizer, epoch, milestones=None):
    """Sets the learning rate: milestone is a list/tuple"""

    def to(epoch):
        if epoch <= args.warmup:
            return 1
        elif args.warmup < epoch <= milestones[0]:
            return 0
        for i in range(1, len(milestones)):
            if milestones[i - 1] < epoch <= milestones[i]:
                return i
        return len(milestones)

    n = to(epoch)

    # global lr
    lr = args.base_lr * (0.2**n)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr