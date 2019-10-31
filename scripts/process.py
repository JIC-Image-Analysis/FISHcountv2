import logging

import numpy as np
import pandas as pd

from dtoolbioimage import Image

from dtoolbioimage import ilogging

from segmentation import segment_fishimage
from probes import find_probe_locations, visualise_probe_locations
from annotation import generate_annotated_image, generate_label_image


def generate_data_frame(segmentation, centroids_tuple, area_scale):

    rids = set(np.unique(segmentation)) - set([0])
    
    region_data = []
    for rid in rids:
        region_positions = set(zip(*np.where(segmentation == rid)))
        region_centroids = region_positions & centroids_tuple
        n_probes = len(region_centroids)
        area = len(region_positions)
        area_microns = int(area * area_scale)

        region_data.append({
            "Cell no": rid,
            "mRNA no": n_probes,
            "Area (µm)": area_microns
        })

    return pd.DataFrame(region_data)



def find_probes_segment_and_create_annotation(fishimage, probe_channel, template, image_output_fpath, csv_output_fpath):

    max_proj = np.max(fishimage.probes[probe_channel], axis=2).view(Image)
    nuclear_proj = np.max(fishimage.nuclei, axis=2).view(Image)
    ilogging.info(max_proj, "probe_channel_max_projection")

    segmentation = segment_fishimage(fishimage)
    ilogging.info(segmentation.pretty_color_image.view(Image), "segmentation")

    centroids = find_probe_locations(fishimage, probe_channel, template)
    logging.info(f"Found {len(centroids)} probes")
    centroids_tuple = set([tuple(c) for c in centroids])
    visualise_probe_locations(max_proj, centroids_tuple)

    physX = float(fishimage.nuclei.metadata.PhysicalSizeX)
    physY = float(fishimage.nuclei.metadata.PhysicalSizeY)
    area_scale = physX * physY

    # annotated_image = generate_annotated_image(max_proj, segmentation, centroids_tuple, area_scale)
    annotated_image = generate_label_image(max_proj, nuclear_proj, segmentation, centroids_tuple, area_scale)
    df = generate_data_frame(segmentation, centroids_tuple, area_scale)

    df.to_csv(csv_output_fpath, index=False)
    annotated_image.save(image_output_fpath)
