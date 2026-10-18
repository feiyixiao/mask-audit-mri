"""Render (image, mask) as a picture a vision-language model can look at:
three orthogonal slices through the mask centre, grayscale MRI, mask outline in red.
"""
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

PANEL = 224  # pixels per panel


def _panel(slice2d, mask2d):
    g = (np.clip(slice2d, 0, 1) * 255).astype(np.uint8)
    rgb = np.stack([g, g, g], axis=-1)
    edge = mask2d & ~ndi.binary_erosion(mask2d)
    rgb[edge] = (255, 0, 0)
    im = Image.fromarray(rgb)
    # nearest-neighbour keeps the one-voxel outline sharp
    return im.resize((PANEL, PANEL), Image.NEAREST)


def render(img, mask):
    if mask.any():
        c = [int(round(x)) for x in ndi.center_of_mass(mask)]
    else:
        c = [s // 2 for s in img.shape]
    panels = [
        _panel(img[c[0], :, :], mask[c[0], :, :]),
        _panel(img[:, c[1], :], mask[:, c[1], :]),
        _panel(img[:, :, c[2]], mask[:, :, c[2]]),
    ]
    out = Image.new("RGB", (PANEL * 3 + 8, PANEL), (0, 0, 0))
    for i, p in enumerate(panels):
        out.paste(p, (i * (PANEL + 4), 0))
    return out
