# This script provides a function to visualize a batch of images with corresponding instance segmentation masks.
# It uses matplotlib to display the images in a grid and to overlay the instance masks.
# The function is useful for analyzing the performance of instance segmentation models.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

def visualize_instance_segmentation(images_list, masks_list, size=6, shape=(10, 10), title=None, save=None):
    """
    Visualize a batch of images with their corresponding instance segmentation masks.
    
    Args:
        images_list (list): A list of images to display.
        masks_list (list): A list of instance segmentation masks for each image.
        size (int, optional): Size of the entire grid figure. Default is 6.
        shape (tuple, optional): Shape of the grid (rows, columns). Default is (10, 10).
        title (str, optional): Title of the grid. Default is None.
        save (str, optional): Path to save the grid figure. Default is None.
    """
    # Create a figure with a specified size.
    fig = plt.figure(figsize=(size, size))
    # Create an ImageGrid with the specified shape and padding.
    grid = ImageGrid(fig, 111, nrows_ncols=shape, axes_pad=0.08)

    # Iterate through images and masks.
    for ax, image, mask in zip(grid, images_list, masks_list):
        # Normalize the image to the range [0, 1].
        image = (image - image.min()) / (image.max() - image.min())
        
        # Overlay the mask on the image.
        overlay = np.where(mask > 0, mask, np.nan)
        
        # Display the image and overlay the mask.
        ax.imshow(image, cmap='gray')
        ax.imshow(overlay, cmap='jet', alpha=0.5)
        ax.axis('off')

    # Display the title if specified.
    if title:
        print(title)
    # Save the figure if specified.
    if save:
        plt.savefig(save)
    # Show the plot.
    plt.show()
