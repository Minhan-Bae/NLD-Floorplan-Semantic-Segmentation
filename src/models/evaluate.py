import sys
sys.path.append("..")
from importmod import *

def compute_miou(predictions, targets):
    batch_size = predictions.size(0)
    num_classes = predictions.size(1)
    miou = 0.0

    for c in range(num_classes):
        intersection = 0.0
        union = 0.0

        for i in range(batch_size):
            pred = (predictions[i, c] > 0.5).int()  # 예측된 클래스 c의 마스크
            target = (targets[i, c] > 0.5).int()  # 실제 클래스 c의 마스크

            intersection += (pred & target).sum().item()
            union += (pred | target).sum().item()

        if union == 0:  # union이 0인 경우 처리
            iou = 0.0
        else:
            iou = intersection / union

        miou += iou

    miou /= num_classes

    return miou

def train_one_epoch(args, epoch, data_loader, model, criterion, optimizer, scheduler, device):
    model.train()
    model.to(device)
    
    cnt = 0
    correct = 0
    scaler = GradScaler()

    pbar = tqdm(enumerate(data_loader), total=len(data_loader))
    for step, (image, label) in pbar:

        image = image.to(device)
        label = label.to(device)

        with autocast(enabled=True):
            model = model.to(device)
            output = model(image)
            
            loss = criterion(output, label)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        optimizer.zero_grad()

        _, preds = torch.max(output, 1)
        preds = preds.unsqueeze(1)  # 두 번째 차원 추가
        correct += torch.sum(preds == label.data)
        cnt += 1

        description = f"| # Epoch : {epoch + 1} Loss : {(loss.item()):.4f}"
        pbar.set_description(description)

    scheduler.step()
    
    msg = (
    "Epoch: {}\t".format(str(epoch).zfill(len(str(args.epochs))))
    + "LR: {:.8f}\t".format(args.base_lr)
    + "Loss: {:.8f}\t".format(loss / len(data_loader))
    )
    return msg


def valid_one_epoch(args, data_loader, model, criterion, device, save=None):
    cum_loss = 0.0
    cum_miou = 0.0  # mIoU를 누적할 변수

    model.eval()
    model = model.to(device)
    with torch.no_grad():
        for features, labels in data_loader:
            features = features.to(device)
            labels = labels.to(device)
            
            with autocast(enabled=True):
                predicts = model(features).to(device)
                loss = criterion(predicts, labels)

            cum_loss += loss.item()
            
            # mIoU 계산
            miou = compute_miou(predicts, labels)
            cum_miou += miou

    if save:
        visualize_instance_segmentation.visualize_instance_segmentation(
            args,
            features,
            predicts,
            save=save,
        )
    
    avg_loss = cum_loss / len(data_loader)
    avg_miou = cum_miou / len(data_loader)

    return avg_loss, avg_miou