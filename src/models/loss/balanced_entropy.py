import torch
import torch.nn.functional as F

def balanced_entropy(preds,targets):
    # Initialization and preprocessing
    eps = 1e-6  # for numerical stability
    num_classes = targets.size(1)  # find the number of classes from the targets
    ind = torch.argmax(targets, 1).type(torch.int)  # get the index of the class with the highest score in the targets
    total = torch.sum(targets)  # compute the total sum of targets
    
    # Apply softmax to the predictions and clip the softmax outputs to prevent division by zero in log computation
    z = torch.clamp(F.softmax(preds, dim=1), min=eps, max=1-eps)
    log_z = torch.log(z)  # compute the log of clipped softmax outputs

    # Compute the count of samples in each class
    m_c = (ind == torch.arange(num_classes)[:, None]).type(torch.int)  # Vectorized the computation of m_c
    n_c = torch.sum(m_c, dim=1).type(torch.float)  # Vectorized the computation of n_c
    
    # Compute the count of samples not in each class and the total count
    c = total - n_c
    tc = c.sum()

    # Compute the weights for each class
    w = c / tc

    # Compute the weighted loss for each class
    loss = 0
    for i in range(num_classes):
        # One-hot encode the mask for each class
        m_c_one_hot = F.one_hot((i * m_c[i]).permute(1, 2, 0).type(torch.long), num_classes=num_classes)
        m_c_one_hot = m_c_one_hot.permute(2, 3, 0, 1)
        
        # Compute the target for each class
        y_c = m_c_one_hot * targets
        
        # Add the weighted loss for each class
        loss += w[i] * (-torch.sum(y_c * log_z, dim=2)).sum()  # Used the pre-computed weight

    # Return the averaged loss
    return loss / num_classes
