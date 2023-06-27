def adjust_learning_rate(args, optimizer, epoch):
    """Sets the learning rate: milestone is a list/tuple"""

    def to(epoch):
        if epoch <= args.warmup:
            return 1
        elif args.warmup < epoch <= args.milestones[0]:
            return 0
        for i in range(1, len(args.milestones)):
            if args.milestones[i - 1] < epoch <= args.milestones[i]:
                return i
        return len(args.milestones)

    n = to(epoch)

    # global lr
    lr = args.base_lr * (0.2**n)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr