"""Count fluorescent probes and assign to cells."""

import click
import logging
import dtoolcore

from dtoolbioimage import ImageDataSet, Image

from dataloader import FISHImage
from probes import make_stage2_template
from process import find_probes_segment_and_create_annotation

logger = logging.getLogger("FISHCountv2")


def extract_template(ids, image_name, series_name, nuclear_channel_first, probe_channel):
    fishimage = FISHImage.from_ids_im_sn(
        ids,
        image_name,
        series_name,
        nuclear_channel_first
    )

    logger.info(f"Loaded image with {len(fishimage.probes)} probe channels")

    return make_stage2_template(fishimage, probe_channel)



@click.command()
@click.argument('ids_uri')
@click.option('--nuclear-channel-first', default=False, is_flag=True)
@click.option('--probe-channel', default=1, type=int)
def main(ids_uri, nuclear_channel_first, probe_channel):

    logging.basicConfig(level=logging.INFO)

    ids = ImageDataSet(ids_uri)

    pairs = ids.get_image_series_name_pairs()

    fca1_pairs = [(s, n) for s, n in pairs if s.startswith("fca1")]
    image_name, series_name = fca1_pairs[2]

    logger.info(f"Processing {image_name}/{series_name}")
    template = extract_template(ids, image_name, series_name, nuclear_channel_first, probe_channel-1)
    template.view(Image).save("template.png")


if __name__ == "__main__":
    main()
