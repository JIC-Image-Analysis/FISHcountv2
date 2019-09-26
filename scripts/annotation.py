import os

import numpy as np

from skimage.segmentation import mark_boundaries

from PIL import Image, ImageDraw, ImageFont

from dtoolbioimage import scale_to_uint8
from dtoolbioimage import Image as dbiImage

HERE = os.path.dirname(__file__)
FONT_PATH = os.path.join(HERE, "../fonts/UbuntuMono-R.ttf")


def generate_annotated_image(max_proj, segmentation, centroids_tuple, area_scale=1):
    border_image = scale_to_uint8(mark_boundaries(scale_to_uint8(max_proj), segmentation, color=(1, 0, 0)))
    annotated = Image.fromarray(border_image)
    draw = ImageDraw.Draw(annotated)
    font_size = 14
    font = ImageFont.truetype(FONT_PATH, font_size)

    rids = set(np.unique(segmentation)) - set([0])
    for rid in rids:
        region_positions = set(zip(*np.where(segmentation == rid)))
        region_centroids = region_positions & centroids_tuple
        n_probes = len(region_centroids)

        r, c = list(map(int, np.array(list(region_positions)).mean(axis=0)))

        area = len(region_positions)
        area_microns = int(area * area_scale)
        area_label = str(area_microns) + 'µm'

        area_label_offset = int(0.5 * font_size * len(area_label) / 2)
        r_offset = font_size/2

        draw.text((c-10, r-r_offset), str(n_probes), fill=(255, 255, 0), font=font)
        draw.text((c-area_label_offset, r+r_offset), area_label, fill=(255, 255, 0), font=font)

    annotated_as_array = np.array(annotated)
    return annotated_as_array.view(dbiImage)
