import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def get_instance_segmentation_model(num_classes):
    # Load a model pre-trained on COCO and get the reference to the output layers
    model = maskrcnn_resnet50_fpn(pretrained=True)

    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256

    # And replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    return model

def main():
    # Define the number of classes for your problem
    num_classes = 2

    # Get the model using our helper function
    model = get_instance_segmentation_model(num_classes)

    # Move model to the device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # For Mask R-CNN, we can't print the summary using torchsummary because the input size is variable
    # Instead, we will try a forward pass with dummy data to make sure the model works
    x = torch.rand(3, 208, 278).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        predictions = model(x)

    # Print input and output shapes
    print(f"Input shape: {x.shape}")
    for idx, prediction in enumerate(predictions):
        print(f"\nOutput shape for prediction {idx + 1}:")
        for key, value in prediction.items():
            if torch.is_tensor(value):
                print(f"{key}: {value.shape}")
            else:
                print(f"{key}: {value}")

if __name__ == "__main__":
    main()
