from dataclasses import dataclass
from typing import List

from dtoolbioimage import Image3D


@dataclass
class FISHImage(object):

    probes: List[Image3D]
    nuclei: Image3D

    @classmethod
    def from_ids_im_sn(cls, ids, image_name, series_name, nuclear_channel_first):
        channels = list(ids.planes_index[image_name][series_name][0].keys())
        n_channels = len(channels)
        n_probe_channels = n_channels - 1

        if nuclear_channel_first:
            nuclei = ids.get_stack(image_name, series_name, 0, 0)
        else:
            nuclei = ids.get_stack(image_name, series_name, 0, n_channels-1)

        probes = []
        for n in range(n_probe_channels):
            probes.append(ids.get_stack(image_name, series_name, 0, n+1))


        return cls(probes, nuclei)
