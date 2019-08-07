"""Count fluorescent probes and assign to cells."""

import click

from ddcache import ids_from_fpath_with_cache

from dataloader import FISHImage
from segmentation import segment_fishimage
from probes import find_probe_locations
from annotation import generate_annotated_image


def ids_from_image_fpath(image_fpath):
    ids = ids_from_fpath_with_cache(image_fpath)

    return ids


@click.command()
@click.argument('image_fpath')
@click.option('--nuclear-channel-first', default=False, is_flag=True)
def main(image_fpath, nuclear_channel_first):
    ids = ids_from_image_fpath(image_fpath)

    image_name, series_name = ids.get_image_series_name_pairs()[0]

    fishimage = FISHImage.from_ids_im_sn(
        ids,
        image_name,
        series_name,
        nuclear_channel_first
    )

    import numpy as np
    from dtoolbioimage import Image
    max_proj = np.max(fishimage.probes[0], axis=2).view(Image)
    np.max(fishimage.nuclei, axis=2).view(Image).save('nuclei.png')

    segmentation = segment_fishimage(fishimage)
    segmentation.pretty_color_image.view(Image).save("pci.png")

    centroids = find_probe_locations(fishimage)
    centroids_tuple = set([tuple(c) for c in centroids])

    print(fishimage.nuclei.metadata.PhysicalSizeX)

    annotated_image = generate_annotated_image(max_proj, segmentation, centroids_tuple)

    annotated_image.save('output.png')




if __name__ == "__main__":
    main()
