import yaml, numpy as np
from PIL import Image
import matplotlib.pyplot as plt 
#import cv2
from scipy import ndimage as ndi
from scipy.ndimage import distance_transform_edt

with open('/ibex/user/alham0a/localization_project/data/datasets/umgloc_dataset/b5l5c2.yaml') as f:
    meta = yaml.safe_load(f)

img = np.array(Image.open(meta['image']))    # [0,1]
img = (255 - img) / 255 # ROS convention occ=1 white (>occ_thresh), free=0 (<free_thresh) black 
def crop(img, meta, roi=None):
    """
    roi = (x0_px, y0_px, x1_px, y1_px)  in pixel coordinates.
    If roi is None, skip cropping.
    """
    if roi is None:
        return img, meta

    x0,y0,x1,y1 = roi
    img_c = img[y0:y1, x0:x1]

    # ---- update origin so trajectories still line up ----
    res = meta['resolution']                # m/px
    ox, oy, oyaw = meta['origin']           # metres, radians
    new_origin = [ox + x0*res, oy + (img.shape[0]-y1)*res, oyaw]
    meta_c = {**meta, 'origin': new_origin}

    return img_c, meta_c

img_c, meta_c = crop(img, meta, roi=(0, 0, -1, 350))
plt.imshow(img_c, cmap='gray')
plt.show()


# Compute the free mask
free_mask = img_c < 0.196  # as specified

# Define processing functions
def apply_erosion(mask, kernel_size=2):
    struct = ndi.iterate_structure(ndi.generate_binary_structure(2, 2), kernel_size)
    return ndi.binary_erosion(mask, structure=struct)

def apply_dilation(mask, kernel_size=2):
    struct = ndi.iterate_structure(ndi.generate_binary_structure(2, 1), kernel_size)
    return ndi.binary_dilation(mask, structure=struct)

def apply_opening(mask, kernel_size=2):
    struct = ndi.iterate_structure(ndi.generate_binary_structure(2, 1), kernel_size)
    return ndi.binary_opening(mask, structure=struct)

def apply_closing(mask, kernel_size=2):
    struct = ndi.iterate_structure(ndi.generate_binary_structure(2, 1), kernel_size)
    return ndi.binary_closing(mask, structure=struct)

def apply_gaussian_smoothing(mask, sigma=1.0):
    return ndi.gaussian_filter(mask.astype(float), sigma=sigma)

def apply_size_filtering(mask, min_size=100):
    labeled_mask, num_labels = ndi.label(mask)
    sizes = np.bincount(labeled_mask.ravel())
    keep = sizes >= min_size
    keep_mask = keep[labeled_mask]
    return keep_mask

def apply_opening_then_closing(mask, kernel_size=2):
    return apply_closing(apply_opening(mask, kernel_size), kernel_size)

def apply_closing_then_opening(mask, kernel_size=2):
    return apply_opening(apply_closing(mask, kernel_size), kernel_size)
# Add updated smoothing + thresholding
def apply_gaussian_smoothing_then_threshold(mask, sigma=1.0, thresh=0.5):
    smoothed = ndi.gaussian_filter(mask.astype(float), sigma=sigma)
    return smoothed > thresh

# Updated size filtering on obstacle complement
def apply_size_filtering_on_obstacles(free_mask, min_size=100):
    occ_mask = ~free_mask
    labeled, num_labels = ndi.label(occ_mask)
    sizes = np.bincount(labeled.ravel())
    keep = sizes >= min_size
    filtered_occ = keep[labeled]
    return ~filtered_occ  # invert back to free space
    
free_mask = apply_gaussian_smoothing_then_threshold(free_mask, sigma=1.0, thresh=0.6)
free_mask = apply_opening_then_closing(free_mask, 2)
# free_mask = apply_gaussian_smoothing_then_threshold(free_mask, sigma=1.0)
free_mask = ndi.distance_transform_edt(free_mask)
plt.imshow(free_mask, cmap='gray')
plt.show()



