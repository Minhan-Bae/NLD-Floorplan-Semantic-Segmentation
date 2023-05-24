# This script provides an AverageMeter class for tracking the average value of a metric over time.
# The class can be used to monitor the progress of training and validation processes.

class AverageMeter(object):
    def __init__(self):
        """
        Initialize an AverageMeter instance.
        """
        self.reset()

    def reset(self):
        """
        Reset the state of the AverageMeter instance.
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Update the AverageMeter instance with a new value.

        Args:
            val (float): The new value to be added.
            n (int, optional): The weight of the new value. Default is 1.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
