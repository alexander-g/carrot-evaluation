import argparse
import glob
import json
import os
import typing as tp

import PIL.Image
import PIL.ImageDraw
import networkx as nx
import numpy as np
import numpy.typing as npt
from numpy.lib.stride_tricks import sliding_window_view
import scipy
import skimage


from src import util, vis



def get_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--outputs', 
        required = True, 
        help     = 'Path to folder with .ring_points.json and .treerings.png files'
    )
    parser.add_argument('--annotations', required=True, help='Glob-like pattern')
    return parser


def main(args:argparse.Namespace):
    file_pairs = collect_output_annotation_pairs(args.outputs, args.annotations)

    if len(file_pairs) == 0:
        print('No files found')
        return


    versions = {
        'versions':{
            'model': util.find_model_version(args.outputs),
            'dataset': util.git_version(os.path.dirname(args.annotations)),
            'evaluation': util.git_version(os.path.dirname(__file__)),
        }
    }
    
    all_metrics = []
    for i, item in enumerate( file_pairs ):
        print(os.path.basename(item.annotation_png))

        metrics, visrgb = \
            evaluate_ring_points(item.ring_points_json, item.annotation_png)
        
        px_metrics, px_visrgb = \
            evaluate_treering_mask(item.treerings_png, item.annotation_png)

        metrics.update(px_metrics)
        all_metrics += [metrics]
        # vis.save_visualization(args.outputs, outputfile, visrgb)

        vis.save_visualization(args.outputs, item.ring_points_json+'.png', visrgb)
        vis.save_visualization(args.outputs, item.treerings_png, px_visrgb)


    filenames  = [os.path.basename(item.annotation_png) for item in file_pairs]
    outputfile = os.path.join(args.outputs, 'metrics.json')
    print('Saving to ', outputfile)
    metrics    = util.save_metrics(
        outputfile, 
        all_metrics, 
        filenames, 
        versions,
        meanmetrics = util.DEFAULT_MEAN_METRICS + ['px-recall', 'px-precision'],
        summetrics  = util.DEFAULT_SUM_METRICS,
    )



class FileTuple(tp.NamedTuple):
    annotation_png:   str
    treerings_png:    str
    ring_points_json: str

def collect_output_annotation_pairs(
    outputfolder: str, 
    annotations:  str,
) -> list[FileTuple]:
    anfiles  = sorted( glob.glob(annotations) )

    ring_points_pattern = os.path.join(outputfolder, '*.ring_points.json')
    ring_points_files   = sorted( glob.glob(ring_points_pattern) )
    treerings_pattern   = os.path.join(outputfolder, '*.treerings.png')
    treerings_files     = sorted( glob.glob(treerings_pattern) )

    assert len(treerings_files) == len(ring_points_files) == len(anfiles), \
        [len(treerings_files), len(ring_points_files), len(anfiles)]

    output:list[FileTuple] = []
    for i in range(len(anfiles)):
        assert os.path.basename(anfiles[i]).split('.')[0] \
            == os.path.basename(treerings_files[i]).split('.')[0] \
            == os.path.basename(ring_points_files[i]).split('.')[0]
        
        output.append(
            FileTuple(anfiles[i], treerings_files[i], ring_points_files[i])
        )
    return output


def evaluate_treering_mask(outputfile:str, annotationfile:str):
    an  = np.array( PIL.Image.open(annotationfile).convert('L') ) > 0
    out = np.array( PIL.Image.open(outputfile).convert('L') ) > 0

    an  = skimage.morphology.skeletonize(an > 0)
    out = skimage.morphology.skeletonize(out > 0)

    an_dilated  = scipy.ndimage.binary_dilation(an,  iterations=30)
    out_dilated = scipy.ndimage.binary_dilation(out, iterations=30)

    recall_map    = an & out_dilated
    precision_map = out & an_dilated

    recall    = recall_map.sum() / an.sum() if an.sum() > 0 else 0.0
    precision = precision_map.sum() / out.sum() if out.sum() > 0  else 0.0
    f1        = util.weighted_harmonic_mean([recall, precision])

    visrgb        = np.zeros(an.shape + (3,), dtype='bool')
    visrgb[...,1] = recall_map
    visrgb[...,2] = (an  & ~recall_map)
    visrgb[...,0] = (out & ~precision_map)

    # dilate for better visibility
    visrgb = scipy.ndimage.binary_dilation(visrgb, iterations=5, axes=[0,1])
    visrgb = (visrgb * 255).astype('uint8')

    return {
        'px-recall':    recall,
        'px-precision': precision,
        'px-hmean':     f1,
    }, visrgb




def evaluate_ring_points(outputfile:str, annotationfile:str):
    an = np.array( PIL.Image.open(annotationfile).convert('L') )
        
    annotation = convert_annotation_to_instancemap(an)
    output     = load_output_as_instancemap(outputfile, an.shape)

    iou     = util.IoU_matrix(annotation, output, zero_out_zero=True)
    matches = util.best_iou_matches(iou, threshold=0.4)
    visrgb  = vis.create_visualization(
        annotation, 
        output, 
        matches.matches0.astype('int32'), 
        matches.matches1.astype('int32')
    )

    TP = len(matches.matches0)
    FP = int(output.max() - TP)
    FN = int(annotation.max() - TP)
    mIoU = float( matches.values.mean() if TP > 0 else 0.0 )

    return {
        'TP':        TP,
        'FP':        FP,
        'FN':        FN,
        'precision': TP / (TP+FP) if (TP+FP) else 0.0,
        'recall':    TP / (TP+FN) if (TP+FN) else 0.0,
        'mIoU':      mIoU,
    }, visrgb


def load_output_as_instancemap(outputfile:str, shape:tuple[int,int]) -> np.ndarray:
    with open(outputfile) as f:
        jsondata = json.load(f)
    
    assert type(jsondata) == list
    paths = [np.array(p) for p in jsondata]
    assert all([p.ndim==3 and p.shape[0] == 2 and p.shape[-1] == 2 for p in paths])

    instancemap = np.zeros(shape, dtype='int32')
    for i, dualpath in enumerate(paths, start=1):
        path0 = dualpath[0]
        path1 = dualpath[1]
        polygon = np.concatenate([path0, path1[::-1]])
        instancemap = np.maximum(instancemap, draw_polygon(polygon, shape) * i)
    return instancemap.astype('int32')


def convert_annotation_to_instancemap(an:np.ndarray) -> np.ndarray:
    assert an.ndim == 2

    sk = skimage.morphology.skeletonize(an > 0)
    sk_labeled, n = scipy.ndimage.label(sk, structure=np.ones([3,3]))

    paths = []
    for i in range(1, n+1):
        G, nodes = skimage.graph.pixel_graph(sk_labeled == i, connectivity=8)
        neighbors_per_node =  np.array( (G > 0).sum(1) ).ravel()
        assert neighbors_per_node.max() == 2
        assert neighbors_per_node.min() == 1
        endpoints = np.argwhere( (neighbors_per_node == 1) ).ravel()
        assert len(endpoints), 'more than 2 endpoints not implemented yet'

        Gnx  = nx.from_scipy_sparse_array(G)
        path = nx.shortest_path(Gnx, source=endpoints[0], target=endpoints[1])
        path_xy = np.argwhere(sk_labeled==i)[:,::-1][path]

        paths.append(path_xy)
    
    pairs = pair_consecutive_convex_paths(paths)
    instancemap = np.zeros(sk.shape, dtype=np.int32)
    for k, [i,j] in enumerate(pairs, start=1):
        polygon = combine_paths_into_polygon(paths[i], paths[j])
        instancemap = np.maximum(instancemap, draw_polygon(polygon, sk.shape) * k)
    return instancemap.astype('int32')


def pair_consecutive_convex_paths(paths: list[np.ndarray]) -> list[tuple[int,int]]:
    pairs = []
    for i, path_i in enumerate(paths):
        for j, path_j in enumerate(paths):
            if i == j:
                continue
            if (j,i) in pairs:
                continue 

            polygon = combine_paths_into_polygon(path_i, path_j)

            # if any other object is inside the polygon, reject
            for k, path_k in enumerate(paths):
                if k in [i,j]:
                    continue

                if points_in_polygon(polygon, path_k).any():
                    # another object is inside, reject
                    break
            else:
                # no other object is inside, valid pair
                pairs.append( (i,j) )
    return pairs


def points_in_polygon(polygon:np.ndarray, points:np.ndarray) -> np.ndarray:
    '''Test if points are inside the polygon, returning a bool aray'''

    assert polygon.ndim == 2
    assert polygon.shape[-1] == 2
    assert points.ndim == 2
    assert points.shape[-1] == 2

    if len(polygon) < 3:
        return np.zeros(len(points), dtype=bool)

    x, y = points[:, 0], points[:, 1]
    inside = np.zeros(len(points), dtype=bool)
    on_boundary = np.zeros(len(points), dtype=bool)

    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        ax, ay = a
        bx, by = b

        cross = (x - ax) * (by - ay) - (y - ay) * (bx - ax)
        on_boundary |= (
            (cross == 0)
            & (x >= min(ax, bx)) & (x <= max(ax, bx))
            & (y >= min(ay, by)) & (y <= max(ay, by))
        )

        if ay != by:
            crosses_ray = (ay > y) != (by > y)
            x_intersect = ax + (y - ay) * (bx - ax) / (by - ay)
            inside ^= crosses_ray & (x < x_intersect)

    return inside | on_boundary


def combine_paths_into_polygon(path0:np.ndarray, path1:np.ndarray) -> np.ndarray:
    assert path0.ndim == path1.ndim == 2
    assert path0.shape[-1] == path1.shape[-1] == 2
    assert len(path0) >= 2
    assert len(path1) >= 2

    distances = scipy.spatial.distance.cdist( path0[-1:], path1[ [0,-1] ] ).ravel()
    if distances[0] > distances[1]:
        path1 = path1[::-1]

    polygon = np.concatenate( [path0, path1] )
    return polygon

def draw_polygon(polygon_xy:np.ndarray, shape:tuple[int,int]) -> np.ndarray:
    canvas = np.zeros(shape, dtype=np.bool_)
    image  = PIL.Image.fromarray(canvas)
    draw   = PIL.ImageDraw.Draw(image)

    draw.polygon(polygon_xy.tolist(), fill=1)
    return np.array(image)



if __name__ == '__main__':
    args = get_argparser().parse_args()
    main(args)



    print('done')
