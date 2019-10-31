"""Count fluorescent probes and assign to cells."""

import click
import logging

from ddcache import ids_from_fpath_with_cache

from dataloader import FISHImage
from process import find_probes_segment_and_create_annotation


def ids_from_image_fpath(image_fpath):
    ids = ids_from_fpath_with_cache(image_fpath)

    return ids


@click.command()
@click.argument('image_fpath')
@click.option('--nuclear-channel-first', default=False, is_flag=True)
@click.option('--probe-channel', default=1, type=int)
def main(image_fpath, nuclear_channel_first, probe_channel):

    logger = logging.getLogger("FISHCountv2")
    logging.basicConfig(level=logging.INFO)

    ids = ids_from_image_fpath(image_fpath)

    image_name, series_name = ids.get_image_series_name_pairs()[0]

    fishimage = FISHImage.from_ids_im_sn(
        ids,
        image_name,
        series_name,
        nuclear_channel_first
    )

    logger.info(f"Loaded image with {len(fishimage.probes)} probe channels")
    logger.info(f"Counting probes for probe channel {probe_channel}")

    template = None
    image_output_fpath = 'counts.png'
    csv_output_fpath = 'counts.csv'
    find_probes_segment_and_create_annotation(
        fishimage,
        probe_channel-1,
        template,
        image_output_fpath,
        csv_output_fpath
    )


if __name__ == "__main__":
    main()
