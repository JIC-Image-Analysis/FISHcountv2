import logging

from pathlib import Path

import click

import numpy as np
from skimage.exposure import equalize_adapthist

from dtoolbioimage import ImageDataSet, Image

from dataloader import FISHImage


def fishimage_to_eq_nuclear_proj(fishimage):
    ks = 16
    nuclear_channel_proj = np.max(fishimage.nuclei, axis=2)
    eq_nuclear_proj = equalize_adapthist(nuclear_channel_proj, kernel_size=(ks, ks))

    return eq_nuclear_proj.view(Image)


def project_and_save(ids, im, sn, output_dirpath):

    logging.info(f"Processing {im}/{sn}")

    fishimage = FISHImage.from_ids_im_sn(
        ids,
        im,
        sn,
        nuclear_channel_first=True,
    )

    output_fpath = output_dirpath / f"{im}-nuclei.png"
    eq_nuclear_proj = fishimage_to_eq_nuclear_proj(fishimage)
    eq_nuclear_proj.save(output_fpath)


@click.command()
@click.argument('ids_uri')
@click.argument('output_dirpath')
def main(ids_uri, output_dirpath):

    logging.basicConfig(level=logging.INFO)

    ids = ImageDataSet(ids_uri)

    output_dirpath = Path(output_dirpath)
    output_dirpath.mkdir(parents=True, exist_ok=True)

    pairs = ids.get_image_series_name_pairs()

    for im, sn in pairs:
        project_and_save(ids, im, sn, output_dirpath)


if __name__ == "__main__":
    main()
