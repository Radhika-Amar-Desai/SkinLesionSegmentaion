# import the necessary packages
import torch
import os
# base path of the dataset
DATASET_PATH = \
    "C:\\Users\\97433\\unet\\SkinLesionSegmentation\\dataset_for_unet\\pytorch_unet\\train"
# define the path to the images and masks dataset
IMAGE_DATASET_PATH = \
    "C:\\Users\\97433\\unet\\SkinLesionSegmentation\\dataset_for_unet\\pytorch_unet\\train\\gradcam_augmented_images"

MASK_DATASET_PATH = \
    "C:\\Users\\97433\\unet\\SkinLesionSegmentation\\dataset_for_unet\\pytorch_unet\\train\\augmented_labels"    
# define the test split
TEST_SPLIT = 0.15
# determine the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEVICE == "cuda" else False

# define the number of channels in the input, number of classes,
# and number of levels in the U-Net model
NUM_CHANNELS = 1
NUM_CLASSES = 1
NUM_LEVELS = 3
# initialize learning rate, number of epochs to train for, and the
# batch size
INIT_LR = 0.001
NUM_EPOCHS = 40
BATCH_SIZE = 64
# define the input image dimensions
INPUT_IMAGE_WIDTH = 128
INPUT_IMAGE_HEIGHT = 128
# define threshold to filter weak predictions
THRESHOLD = 0.5
# define the path to the base output directory
BASE_OUTPUT = "C:\\Users\\97433\\unet\\SkinLesionSegmentation\\dataset_for_unet\\pytorch_unet\\output"
# define the path to the output serialized model, model training
# plot, and testing image paths
MODEL_PATH = os.path.join(BASE_OUTPUT, "unet_tgs_salt.pth")
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])
TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "test_paths.txt"])
