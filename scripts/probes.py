from skimage.measure import label, regionprops
import numpy as np

from skimage.draw import circle_perimeter
from skimage.feature import match_template as _match_template
from skimage.filters import sobel
from skimage.morphology import disk

from dtoolbioimage import scale_to_uint8, Image
from dtoolbioimage import ilogging

from dtoolbioimage.transformation import create_transformation


@create_transformation
def find_edges(ndarray):
    return sobel(ndarray)


@create_transformation
def match_template(*args, **kwargs):
    return _match_template(*args, **kwargs)


def make_stage1_template():
    """Make a template for initial matching. This is an annulus."""

    template = disk(3)
    template[3, 3] = 0

    return template


def make_stage2_template(fishimage, n_probe=0):
    template = make_stage1_template()
    max_proj = np.max(fishimage.probes[n_probe], axis=2)
    edge_image = find_edges(max_proj)

    match_result = match_template(edge_image, template, pad_input=True)
    cmax = np.max(match_result)

    px, py = list(zip(*np.where(match_result == cmax)))[0]

    tr = 4
    better_template = edge_image[px-tr:px+tr,py-tr:py+tr]

    return better_template


def find_probe_locations(fishimage, n_probe=0):

    template = make_stage2_template(fishimage)
    max_proj = np.max(fishimage.probes[n_probe], axis=2)
    edge_image = find_edges(max_proj)
    match_result = match_template(edge_image, template, pad_input=True)
    overall_points = (match_result > 0.6)
    rprops = regionprops(label(overall_points))
    centroids = [list(map(int, r.centroid)) for r in rprops]

    return centroids

@create_transformation
def visualise_probe_locations(max_proj, centroids):
    canvas = np.dstack(3 * [scale_to_uint8(max_proj)])

    for r, c in centroids:
        rr, cc = circle_perimeter(r, c, 3)
        try:
            canvas[rr, cc] = 255, 255, 0
        except IndexError:
            pass

    return canvas.view(Image)
