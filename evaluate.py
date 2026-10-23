import argparse
import glob
import json
import os
import typing as tp

import numpy as np
import PIL.Image
import scipy

from src import util, vis



def get_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputs', required=True, help='Path to outputs')
    parser.add_argument('--annotations', required=True, help='Path to split file')
    return parser


def main(args:argparse.Namespace):
    versions = {
        'versions':{
            'model': util.find_model_version(os.path.dirname(args.outputs)),
            'dataset': util.git_version(os.path.dirname(args.annotations)),
            'evaluation': util.git_version('./'),
        }
    }

    file_pairs = collect_output_annotation_pairs(
        args.outputs, 
        args.annotations, 
        '.png'
    )
    if len(file_pairs) == 0:
        print('No matching files found.')
        return
    
    all_metrics = []
    for i, (annotationfile, outputfile) in enumerate(file_pairs):
        print(f'[{i:3d}/{len(file_pairs)}] {os.path.basename(annotationfile)}')
        metrics, visrgb = evaluate_file_pair(outputfile, annotationfile)
        all_metrics += [metrics]
        vis.save_visualization(args.outputs, outputfile, visrgb)
    outputfile = os.path.join(args.outputs, 'metrics.json')
    filenames  = [os.path.basename(a) for a,_ in file_pairs]
    metrics = save_metrics(outputfile, all_metrics, filenames, versions)
    # print('merged metrics:', metrics)


def collect_output_annotation_pairs(
    outputfolder:   str, 
    annotationfile: str,
    file_ending:    str,
) -> tp.List[tp.Tuple[str,str]]:
    pattern = os.path.join(outputfolder, f'*{file_ending}')
    outputfiles = sorted(glob.glob( pattern ))
    input_annotation_pairs = util.load_file_tuples(annotationfile, ',', 2)
    annotationfiles = [af for _,af in input_annotation_pairs]

    output_annotation_pairs = {
        af:of for of in outputfiles for af in annotationfiles 
            if are_files_matching(of, af, file_ending, '.png')
    }
    
    missing_outputfiles = [
        os.path.basename(of) for of in outputfiles 
            if of not in output_annotation_pairs.values()
    ]
    missing_annotationfiles = [
        os.path.basename(af) for af in annotationfiles 
            if af not in output_annotation_pairs.keys()
    ]
    if len(missing_outputfiles):
        print('Unmatched outputfiles:',     missing_outputfiles)
    if len(missing_annotationfiles):
        print('Unmatched annotationfiles:', missing_annotationfiles)
    return list(output_annotation_pairs.items())


def are_files_matching(
    outputfile:str, 
    annotationfile:str,
    output_file_ending:str,
    annotation_file_ending:str,
) -> bool:
    outputfile = \
        os.path.basename(outputfile).replace(output_file_ending, '')
    annotationfile = \
        os.path.basename(annotationfile).replace(annotation_file_ending, '')
    return outputfile == annotationfile[:len(outputfile)]


def evaluate_file_pair(outputfile:str, annotationfile:str):
    output = util.load_image(outputfile, mode='L')
    output, _ = scipy.ndimage.label( output > 0.5, np.ones([3,3]) )

    annotation = util.load_image(annotationfile, mode='L')
    annotation, _ = scipy.ndimage.label(annotation > 0.5, np.ones([3,3]) )

    iou = util.IoU_matrix(annotation, output, zero_out_zero=True)
    matches = util.best_iou_matches(iou, threshold=0.7)
    visrgb = vis.create_visualization(
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
        'precision': TP / (TP+FP),
        'recall':    TP / (TP+FN),
        'mIoU':      mIoU,
    }, visrgb

def save_metrics(
    outputfile: str,
    metrics:    tp.List[tp.Dict[str,float]], 
    files:      tp.List[str], 
    extras:     tp.Dict|None = None
):
    jsondata = {} # type: ignore [var-annotated]
    if extras is not None:
        jsondata = jsondata | extras
    jsondata = jsondata | merge_metrics(metrics)
    for m,f in zip(metrics, files):
        jsondata[f] = m
    open(outputfile, 'w').write(json.dumps(jsondata, indent=2))
    return jsondata['mean']


def merge_metrics(metrics:tp.List[tp.Dict[str,float]]) -> tp.Dict:
    if len(metrics) == 0:
        return {}
    
    meankeys = ['precision', 'recall', 'mIoU']
    sumkeys  = ['TP', 'FP', 'FN']
    meanmetrics = {k:float(np.mean( [m[k]  for m in metrics] )) for k in meankeys}
    meanmetrics = meanmetrics | {
        'hmean': util.weighted_harmonic_mean([v for v in meanmetrics.values()])
    }

    summetrics  = {k:int(np.sum( [m[k]  for m in metrics] )) for k in sumkeys}
    summetrics  = summetrics | { # type: ignore
        'precision': summetrics['TP'] / max(summetrics['TP'] + summetrics['FP'], 1),
        'recall': summetrics['TP'] / max(summetrics['TP'] + summetrics['FN'], 1),
    }
    return {
        'mean': meanmetrics,
        'sum':  summetrics,
    }





if __name__ == '__main__':
    args = get_argparser().parse_args()
    main(args)
    print('done')

