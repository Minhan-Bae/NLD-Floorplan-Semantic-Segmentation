import time
from tqdm import tqdm

import torch
from utils.average_meter import AverageMeter

import sys
sys.path.append('..')
from loss import balanced_entropy as BE
from pytorch_toolbelt import losses as PL
from utils.visualize_instance_segmentation import visualize_instance_segmentation
def train_one_epoch(args, train_loader, model, optimizer, epoch, lr, device, savePath):
    """Network training, loss updates, and backward calculation"""

    # AverageMeter for statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_loss = 0.0

    # criterion = L.FocalLoss()
    
    end = time.time()
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))        
    for idx, (features, labels) in pbar:
        optimizer.zero_grad()
        model.train()
        model = model.to(device)
        
        features = features.to(device)
        labels = labels.to(device)

        predicts = model(features).to(device)

        loss = PL.balanced_binary_cross_entropy_with_logits(predicts, labels)
        
        data_time.update(time.time() - end)
        loss.backward()
        optimizer.step()

        optimizer.zero_grad()
        flag, _ = optimizer.step_handleNan()

        if flag:
            print(
                "Nan encounter! Backward gradient error. Not updating the associated gradients."
            )

        train_loss += loss.item()
        batch_time.update(time.time() - end)
        end = time.time()
    msg = (
        "Epoch: {}\t".format(str(epoch).zfill(len(str(args.epochs))))
        + "LR: {:.8f}\t".format(lr)
        + "Time: {:.3f} ({:.3f})\t".format(batch_time.val, batch_time.avg)
        + "Loss: {:.8f}\t".format(train_loss / len(train_loader))
    )
    return msg
    
def valid_one_epoch(valid_loader, model, device, save=None):
    cum_loss = 0.0
    cum_nme = 0.0

    model.eval()

    with torch.no_grad():
        for features, labels in valid_loader:
            features = features.to(device)
            labels = labels.to(device)

            predicts = model(features).to(device)

            loss = PL.balanced_binary_cross_entropy_with_logits(predicts, labels)
            # nme = NME(outputs, labels)

            # cum_nme += nme.item()
            cum_loss += loss.item()
            break
    if save:
        visualize_instance_segmentation(
            features[:16].cpu(),
            predicts[:16].cpu(),
            labels[:16].cpu(),
            shape=(4, 4),
            size=16,
            save=save,
        )

    return cum_loss / len(valid_loader)