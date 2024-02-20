import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

import config
from dataset import SegmentationDataset
import os
from imutils import paths

if __name__ == "__main__":

    def calculate_iou(pred_mask, true_mask):
        intersection = torch.logical_and(pred_mask, true_mask).sum()
        union = torch.logical_or(pred_mask, true_mask).sum()
        iou = intersection / union.float()
        return iou.item()

    def calculate_dice(pred_mask, true_mask):
        intersection = torch.logical_and(pred_mask, true_mask).sum()
        dice_coefficient = (2.0 * intersection) / (pred_mask.sum() + true_mask.sum())
        return dice_coefficient.item()

    transforms = transforms.Compose([transforms.ToPILImage(),
        transforms.Resize((config.INPUT_IMAGE_HEIGHT,
            config.INPUT_IMAGE_WIDTH)),
        transforms.ToTensor()])

    testImages = sorted(list(paths.list_images(config.IMAGE_DATASET_PATH)))
    testMasks = sorted(list(paths.list_images(config.MASK_DATASET_PATH)))

    validation_dataset = SegmentationDataset(
        imagePaths=testImages, 
        maskPaths=testMasks,
        transforms=transforms)

    validation_loader = DataLoader( validation_dataset , shuffle=False,
        batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
        num_workers=os.cpu_count())

    unet = torch.load(config.MODEL_PATH).to(config.DEVICE)

    unet.eval()
    total_iou = 0.0
    total_dice = 0.0
    num_samples = len(validation_dataset)

    print ( num_samples )

    with torch.no_grad():
        for inputs, targets in validation_loader:
            # Assuming your model outputs segmentation masks
            outputs = unet(inputs)

            # Convert logits to binary masks using a threshold or softmax
            pred_masks = (torch.sigmoid(outputs) > 0.5).float()

            # Assuming targets are binary masks as well
            true_masks = targets.float()

            # Calculate IoU and Dice for each sample
            for i in range(len(inputs)):
                iou = calculate_iou(pred_masks[i], true_masks[i])
                dice = calculate_dice(pred_masks[i], true_masks[i])

                total_iou += iou
                total_dice += dice

                print ( i )

    # Calculate average IoU and Dice across all samples
    average_iou = total_iou / num_samples
    average_dice = total_dice / num_samples

    print(f'Average IoU: {average_iou:.4f}')
    print(f'Average Dice Coefficient: {average_dice:.4f}')
