import os

import numpy as np
import PIL.Image



def create_visualization(
    annotation:  np.ndarray, 
    output:      np.ndarray, 
    an_matches:  np.ndarray,
    out_matches: np.ndarray,
) -> np.ndarray:
    assert annotation.dtype == output.dtype == np.int32
    assert an_matches.dtype == out_matches.dtype == np.int32
    assert annotation.shape == output.shape
    assert an_matches.shape == out_matches.shape
    assert annotation.ndim == 2
    assert an_matches.ndim == 1

    annotation_hits   = np.isin( annotation, an_matches )
    output_hits       = np.isin( output, out_matches )
    annotation_misses = (~annotation_hits) * (annotation > 0)
    output_misses     = (~output_hits) * (output > 0)

    TP_pixels = output_hits & annotation_hits
    FP_pixels = (~TP_pixels) & output_hits
    FN_pixels = (~TP_pixels) & annotation_hits

    RGB = np.zeros(annotation.shape + (3,), dtype='uint8')
    RGB[TP_pixels]                = (0, 255, 0)   # green
    RGB[...,0][output_misses]     = 255           # red   (with potential overlaps)
    RGB[...,2][annotation_misses] = 255           # blue  (with potential overlaps)
    RGB[FP_pixels]                = (255, 255, 0) # yellow
    RGB[FN_pixels]                = (0, 255, 255) # cyan
    return RGB


def save_visualization(outputdir:str, filename:str, rgb:np.ndarray):
    assert rgb.ndim == 3
    assert rgb.shape[-1] == 3
    assert rgb.dtype == np.uint8

    outputdir = os.path.join(outputdir, 'vis')
    os.makedirs(outputdir, exist_ok=True)
    savepath = os.path.join(outputdir, os.path.basename(filename))
    PIL.Image.fromarray(rgb).save(savepath)


