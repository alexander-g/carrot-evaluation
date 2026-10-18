import os
import typing as tp

import numpy as np
import PIL.Image
import scipy


FileTuple = tp.Tuple[str,...]


def load_file_tuples(filepath:str, delimiter:str, n:int) -> tp.List[FileTuple]:
    '''Load n-tuples of file paths from a csv file and check that they exist.'''
    lines = open(filepath, 'r').read().strip().split('\n')
    dirname = os.path.dirname(filepath)
    pairs:tp.List[FileTuple] = []
    for line in lines:
        pair = [f.strip() for f in line.split(delimiter)]
        if len(pair) != n:
            raise Exception(
                f'File does not contain {n}-tuples delimited by "{delimiter}"'
            )
        #convert relative paths to absolute, starting from the textfile directory
        pair = [f if os.path.isabs(f) else os.path.join(dirname, f) for f in pair]
        files_not_found = [f for f in pair if not os.path.exists(f)]
        if len(files_not_found):
            raise Exception(
                f'Files not found: {files_not_found}'
            )
        pairs.append(tuple(pair))
    return pairs


def load_image(
    imagefile: str, 
    normalize: bool = False,
    mode:      str  = 'RGB'
) -> np.ndarray:
    '''Convenience function for loading image files'''
    image  = PIL.Image.open(imagefile).convert(mode)
    pixels = np.array(image)
    if normalize and pixels.dtype == np.uint8:
        pixels = pixels / 255
    return pixels


def IoU_matrix(a:np.ndarray, b:np.ndarray, zero_out_zero:bool) -> np.ndarray:
    # assuming instance values are sequential

    assert a.ndim == b.ndim == 2
    assert a.shape == b.shape
    assert a.dtype == b.dtype
    assert a.dtype in [np.int32, np.int64]

    a = a.astype(np.int32)
    b = b.astype(np.int32)

    uniques_a, counts_a = np.unique(a, return_counts=True)
    uniques_b, counts_b = np.unique(b, return_counts=True)
    matrix = np.empty([len(uniques_a), len(uniques_b)], dtype=np.int64)

    # stack and view as int64 for faster processing
    stacked = np.stack([a,b], axis=-1).view(np.int64)
    uniques_ab, counts_ab = np.unique(stacked, return_counts=True)

    # back to int32 and labels that overlap are paired [N,2]
    unstacked = uniques_ab.view(np.int32).reshape(-1,2)

    intersections_mat = np.zeros_like(matrix)
    intersections_mat[unstacked[:,0], unstacked[:,1]] = counts_ab

    unions_mat =  np.zeros_like(matrix)
    unions_mat += counts_a[:,None]
    unions_mat += counts_b[None,:]
    unions_mat =  unions_mat - intersections_mat

    iou_mat = intersections_mat / unions_mat
    iou_mat[unions_mat == 0] = 0

    if zero_out_zero:
        iou_mat[uniques_a == 0, :] = 0
        iou_mat[:, uniques_b == 0] = 0

    return iou_mat


class IoU_Matches(tp.NamedTuple):
    matches0: np.ndarray
    matches1: np.ndarray
    values:   np.ndarray

def best_iou_matches(
    ioumatrix: np.ndarray, 
    threshold: float
) -> IoU_Matches:
    
    valid    = ioumatrix >= threshold
    filtered = np.where(valid, ioumatrix, -np.inf)

    matches0, matches1 = \
        scipy.optimize.linear_sum_assignment(ioumatrix, maximize=True)

    keep = valid[matches0, matches1]
    matches0, matches1 = matches0[keep], matches1[keep]
    values = ioumatrix[matches0, matches1]

    return IoU_Matches(matches0, matches1, values)



def weighted_harmonic_mean(
    values:  np.ndarray|list[float], 
    weights: np.ndarray|None=None
) -> float:
    values = np.asarray(values, dtype=float)
    if weights is None:
        weights = np.ones_like(values)
    else:
        weights = np.asarray(weights, dtype=float)

    return np.sum(weights) / np.sum(weights / values)

