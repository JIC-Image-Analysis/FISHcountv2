import numpy as np

from skimage.segmentation import mark_boundaries

from PIL import Image, ImageDraw, ImageFont

from dtoolbioimage import scale_to_uint8
from dtoolbioimage import Image as dbiImage


def generate_annotated_image(max_proj, segmentation, centroids_tuple):
    border_image = scale_to_uint8(mark_boundaries(scale_to_uint8(max_proj), segmentation, color=(1, 0, 0)))
    annotated = Image.fromarray(border_image)
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.truetype("UbuntuMono-R.ttf", 20)

    rids = set(np.unique(segmentation)) - set([0])
    for rid in rids:
        region_positions = set(zip(*np.where(segmentation == rid)))
        region_centroids = region_positions & centroids_tuple
        r, c = list(map(int, np.array(list(region_centroids)).mean(axis=0)))
        n_probes = len(region_centroids)
        area = len(region_positions)

        draw.text((c-10, r-10), str(n_probes), fill=(255, 255, 0), font=font)
        draw.text((c-25, r+10), str(area), fill=(255, 255, 0), font=font)

    annotated_as_array = np.array(annotated)
    return annotated_as_array.view(dbiImage)
