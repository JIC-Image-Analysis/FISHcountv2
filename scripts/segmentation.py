
import numpy as np

from scipy.ndimage import distance_transform_edt

from skimage.filters import (
    gaussian,
    threshold_otsu,
    threshold_local
)

from skimage.exposure import equalize_adapthist
from skimage.morphology import remove_small_holes
from skimage.morphology import (
    remove_small_objects,
    label
)

from skimage.segmentation import watershed

from dtoolbioimage.segment import Segmentation


import ilogging
from transformation import create_transformation


equalize_adapthist = create_transformation(equalize_adapthist)


def norm(im):
    return (im - im.min()) / (im.max() - im.min())


@create_transformation
def nuclear_stack_to_segmentation_seeds(nuclear_stack):
    ks = 16  # adaptive histogram kernel size
    bs = 151  # local thresholding block size
    sigma = 9  # smoothing parameter
    min_size = 1000

    nuclear_channel_proj = np.max(nuclear_stack, axis=2)
    eq_nuclear_proj = equalize_adapthist(nuclear_channel_proj, kernel_size=(ks, ks))
    eq_nuclear_proj_smoothed = norm(gaussian(eq_nuclear_proj, sigma=sigma))
    thresh_image = threshold_local(eq_nuclear_proj_smoothed, block_size=bs)
    clipped_thresh_image = np.clip(thresh_image + 0.05, a_min=0, a_max=1)
    base_seeds = (eq_nuclear_proj_smoothed > clipped_thresh_image)

    seeds = remove_small_objects(base_seeds, min_size=min_size)

    return label(seeds)


def segmentation_from_autofluorescence(af_stack, seed_image):

    gaussian_sigma = 10

    min_proj = np.min(af_stack, axis=2)
    smoothed = gaussian(min_proj, sigma=gaussian_sigma)
    mask = (smoothed > threshold_otsu(smoothed))
    distance = distance_transform_edt(mask)
    segmentation = watershed(-distance, seed_image, mask=mask)

    return segmentation.view(Segmentation)


def segment_fishimage(fishimage, n_probe=0):

    seeds = nuclear_stack_to_segmentation_seeds(fishimage.nuclei)
    segmentation = segmentation_from_autofluorescence(fishimage.probes[n_probe], seeds)

    return segmentation

