"""Count fluorescent probes and assign to cells."""

import click
import logging
import dtoolcore

from dtoolbioimage import ImageDataSet, Image

from dataloader import FISHImage
from process import find_probes_segment_and_create_annotation

logger = logging.getLogger("FISHCountv2")


def process_pair(ids, image_name, series_name, nuclear_channel_first, probe_channel, template):
    fishimage = FISHImage.from_ids_im_sn(
        ids,
        image_name,
        series_name,
        nuclear_channel_first
    )

    logger.info(f"Loaded image with {len(fishimage.probes)} probe channels")
    logger.info(f"Counting probes for probe channel {probe_channel}")

    output_fpath = f"{image_name}.png"
    find_probes_segment_and_create_annotation(fishimage, probe_channel-1, template, output_fpath)



@click.command()
@click.argument('ids_uri')
@click.option('--nuclear-channel-first', default=False, is_flag=True)
@click.option('--probe-channel', default=1, type=int)
@click.option('--template-path', default=None)
def main(ids_uri, nuclear_channel_first, probe_channel, template_path):

    logging.basicConfig(level=logging.INFO)

    ids = ImageDataSet(ids_uri)

    pairs = ids.get_image_series_name_pairs()

    fca3_pairs = [(s, n) for s, n in pairs if s.startswith("Ler")]

    if template_path is not None:
        template = Image.from_file(template_path)
    else:
        template=None

    for image_name, series_name in fca3_pairs:
        logger.info(f"Processing {image_name}/{series_name}")
        process_pair(ids, image_name, series_name, nuclear_channel_first, probe_channel, template)


if __name__ == "__main__":
    main()
