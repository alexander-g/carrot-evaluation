import glob
import os

import PIL.Image
import numpy as np
import scipy



anfilesdir = '/home/ubuntu/Projects/CARROT-repos/data/angiosperm-treerings-validation-set/annotations'

anfiles = sorted( glob.glob(anfilesdir + '/*.png') )
assert len(anfiles) > 0


type ConvexHull = scipy.spatial.ConvexHull

def inside_hull(points:np.ndarray, hull:ConvexHull, eps=1e-12) -> np.ndarray:
    assert points.ndim == 2
    assert points.shape[-1] == 2

    inside = np.all(
        points @ hull.equations[:, :-1].T + hull.equations[:, -1] <= 1e-12,
        axis=1
    )
    return inside


def compute_convex_hulls_of_objects(labeled_image:np.ndarray) -> list[ConvexHull]:
    assert labeled_image.ndim == 2
    assert labeled_image.dtype in [np.int32, np.int64]

    hulls:list[ConvexHull] = []
    for l in np.unique(labeled_image):
        if l == 0:
            continue
        
        yx = np.stack( (labeled_image == l).nonzero(), -1 )
        hull = scipy.spatial.ConvexHull(yx)
        hulls.append(hull)
    return hulls


def combine_convex_hulls(hull0:ConvexHull, hull1:ConvexHull) -> ConvexHull:
    yx_0 = hull0.points[hull0.vertices]
    yx_1 = hull1.points[hull1.vertices]
    yx_combined = np.concatenate([yx_0, yx_1])
    return scipy.spatial.ConvexHull(yx_combined)



def pair_consecutive_convex_hulls(hulls: list[ConvexHull]):
    pairs = []
    for i, hull_i in enumerate(hulls):
        for j, hull_j in enumerate(hulls):
            if i == j:
                continue
            if (j,i) in pairs:
                continue 

            hull_combined = combine_convex_hulls(hull_i, hull_j)

            # if any other object is inside the combined hull, reject
            for k, hull_k in enumerate(hulls):
                if k in [i,j]:
                    continue

                yx_k = hull_k.points[hull_k.vertices]
                if inside_hull(yx_k, hull_combined).any():
                    # another object is inside, reject
                    break
            else:
                # no other object is inside, valid pair
                pairs.append( (i,j) )
    return pairs


def chain_sort(indices:list[tuple[int,int]]) -> list[tuple[int,int]] | RuntimeError:
    '''Sort the indices so that output[i][1] == output[i+1][0]'''
    if len(indices) == 0:
        return RuntimeError('Empty list, nothing to sort')

    # copy
    indices = list(indices)

    output:list[tuple[int,int]] = [ indices.pop(0) ]

    while len(indices):
        i,j = output[-1]

        hits = (np.array(indices) == j).any(-1).nonzero()[0]
        if len(hits) != 1:
            return RuntimeError('Discontinuous chain')

        hit = hits[0]
        next_pair = indices.pop(hit)
        if j == next_pair[1]:
            # reverse
            next_pair = next_pair[::-1]
        output.append( next_pair )
    return output


def draw_hull(hull: ConvexHull, shape:tuple[int,int]) -> np.ndarray:
    canvas = np.zeros(shape, dtype=np.bool_)



    return ...


for anf in anfiles:
    an = np.array( PIL.Image.open(anf).convert('L') )
    an_labeled, n_labels = scipy.ndimage.label(an, structure=np.ones([3,3]))
    print(n_labels)

    hulls = compute_convex_hulls_of_objects(an_labeled)
    pairs = pair_consecutive_convex_hulls(hulls)

    chain_sort(hulls[0].simplices)
    breakpoint()


print('done')
