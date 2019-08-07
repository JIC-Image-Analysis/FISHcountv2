import numpy as np
from dtoolbioimage import Image

import ilogging

from segmentation import segment_fishimage
from probes import find_probe_locations, visualise_probe_locations
from annotation import generate_annotated_image



def find_probes_segment_and_create_annotation(fishimage, probe_channel):

    max_proj = np.max(fishimage.probes[probe_channel], axis=2).view(Image)
    ilogging.info(max_proj, "probe_channel_max_projection")

    segmentation = segment_fishimage(fishimage)
    ilogging.info(segmentation.pretty_color_image.view(Image), "segmentation")

    centroids = find_probe_locations(fishimage)
    centroids_tuple = set([tuple(c) for c in centroids])
    visualise_probe_locations(max_proj, centroids_tuple)

    physX = float(fishimage.nuclei.metadata.PhysicalSizeX)
    physY = float(fishimage.nuclei.metadata.PhysicalSizeY)
    area_scale = physX * physY

    annotated_image = generate_annotated_image(max_proj, segmentation, centroids_tuple, area_scale)

    annotated_image.save('output.png')

