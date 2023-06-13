import torch
import torch.nn as nn
import torch.nn.functional as F

def balanced_entropy(preds,targets):
    eps = 1e-6
    m = nn.Softmax(dim=1)
    z = m(preds)
    cliped_z = torch.clamp(z,eps,1-eps)
    log_z = torch.log(cliped_z)
    num_classes = targets.size(1)
    ind = torch.argmax(targets,1).type(torch.int)

    total = torch.sum(targets)
    
    m_c,n_c = [],[]
    for c in range(num_classes):
        m_c.append((ind==c).type(torch.int))
        n_c.append(torch.sum(m_c[-1]).type(torch.float))

    c = []
    for i in range(num_classes):
        c.append(total-n_c[i])
    tc = sum(c)

    loss = 0
    for i in range(num_classes):
        w = c[i]/tc
        m_c_one_hot = F.one_hot(m_c[i].unsqueeze(1).type(torch.long), num_classes)
        m_c_one_hot = m_c_one_hot.permute(0, 2, 1)

        y_c = m_c_one_hot*targets
        loss += w*torch.sum(-torch.sum(y_c*log_z,axis=2))
    return loss/num_classes
