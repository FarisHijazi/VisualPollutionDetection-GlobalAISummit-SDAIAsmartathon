"""
convert yolo format to smartathon stupid format
read coco dir txt files and pandas dataframe and puts them in a csv file ready to submit

repositories/yolov5/runs/detect/exp12/labels
"""
import argparse
import glob
import itertools
import multiprocessing as mp
import os

import pandas as pd
from tqdm import tqdm

import bbox


def cocodir2df(label_paths: list):
    # label_paths: list of paths to coco .txt files
    df_output = pd.DataFrame(columns=['class', 'image_path', 'x', 'y', 'w', 'h'])

    with mp.Pool(processes=mp.cpu_count()) as pool:
        df_list = pool.map(cocodir2df_helper, tqdm(label_paths))
    df_output = pd.concat(df_list)

    return df_output


def cocodir2df_helper(label_path):
    with open(label_path, 'r') as f:
        lines = f.readlines()
    lines = [line.strip().split(' ') for line in lines]
    df_temp = pd.DataFrame(columns=['class', 'image_path', 'x', 'y', 'w', 'h'])
    for line in lines:
        try:
            if len(line) == 6:
                cls, x, y, w, h, conf = line
            elif len(line) == 5:
                cls, x, y, w, h = line
                conf = 1

            df_temp = df_temp.append(
                {
                    'class': float(cls),
                    'image_path': os.path.basename(label_path).replace('.txt', '.jpg'),
                    'x': float(x),
                    'y': float(y),
                    'w': float(w),
                    'h': float(h),
                    'conf': conf,
                },
                ignore_index=True,
            )
        except Exception as e:
            print(e)
            print('error in ', label_path)
            print('line: ', line)

    if not lines:
        # dummy value so that submission is accepted
        print('empty value for ', label_path, 'adding dummy value')
        df_temp = df_temp.append(
            {
                'class': 0.0,
                'image_path': os.path.basename(label_path).replace('.txt', '.jpg'),
                'x': 0,
                'y': 0,
                'w': 0,
                'h': 0,
                'conf': 0,
            },
            ignore_index=True,
        )
    return df_temp


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('labelsdir', nargs='+', help='labels directory containing txt files')
    parser.add_argument('-d', '--data', default='data/Challenge1/', help='data dir')
    parser.add_argument('--conf_thresh', type=float, default=None, help='confidence threshold (before topk)')
    topk = parser.add_mutually_exclusive_group()
    topk.add_argument('--topk', type=int, default=0, help='topk (exact), 0 for all')
    # another type of topk is where we take "at least" topk,
    topk.add_argument('--topk_atleast', type=int, default=0, help='chooses "at least" topk, 0 for all')
    parser.add_argument('--assert_valid', action='store_true', help='assert valid, will fail and exit if not valid')
    overrides = parser.add_argument_group('overrides')
    overrides.add_argument('--setclass', type=int, default=None, help='set all classes to this value')
    overrides.add_argument('--oversized', action='store_true', help='make all image boxes huge')
    args = parser.parse_args()

    assert args.topk == 0 or args.topk_atleast == 0, 'cannot use both topk and topk_atleast'
    if args.topk_atleast > 0:
        assert args.conf_thresh is not None, 'must specify conf_thresh with topk_atleast'

    df = pd.read_csv(args.data + '/train.csv')
    id2name = dict(list(zip(df['class'], df['name'])))

    df_test = pd.read_csv(args.data + '/test.csv')
    df_test['class'] = 3
    df_test['name'] = 'GARBAGE'
    df_test['xmin'] = 1
    df_test['ymin'] = 1
    df_test['xmax'] = 50
    df_test['ymax'] = 50
    df_test = df_test[['class', 'image_path', 'name', 'xmax', 'xmin', 'ymax', 'ymin']]

    args.labelsdir = list(itertools.chain.from_iterable(glob.glob(f) for f in args.labelsdir))
    print('reading from', len(args.labelsdir), 'labelsdir(s)')

    label_paths = list(
        itertools.chain.from_iterable(glob.glob(os.path.join(labelsdir, '*.txt')) for labelsdir in args.labelsdir)
    )
    df_output = cocodir2df(label_paths).astype(
        {
            'class': int,
            'image_path': str,
            'x': float,
            'y': float,
            'w': float,
            'h': float,
            'conf': float,
        }
    )
    df_output['name'] = df_output['class'].apply(lambda x: id2name[int(x)])
    print('total bboxes read:', df_output.shape[0])

    ## convert coco format to smartathon format
    df_output[['xmin', 'ymin', 'xmax', 'ymax']] = (
        df_output[['x', 'y', 'w', 'h']].astype(float).apply(bbox.coco2smartathon, axis=1, result_type='expand')
    )

    # remove items bellow threshold
    if args.conf_thresh and not args.topk_atleast:
        df_output = df_output[df_output['conf'] > args.conf_thresh]
        print('total bboxes after conf_thresh:', df_output.shape[0])

    # groupby image_path and sort by "conf" and choose best confidence row
    df_output = df_output.sort_values(by=['image_path', 'conf'], ascending=False).groupby('image_path')
    if args.topk > 0:
        df_output = df_output.head(args.topk)
        print(f'total bboxes after topk:{args.topk}', df_output.shape[0])
    elif args.topk_atleast > 0:
        # chooses "at least" args.topk_atleast from each image that crosses the threshold
        # this is useful when we want to choose at least 1 item from each image
        df_output = pd.concat(
            [
                df_output.head(args.topk_atleast) if df_output['conf'].max() > args.conf_thresh else df_output.head(0)
                for _, df_output in df_output
            ]
        )
        print(f'total bboxes after topk_atleast:{args.topk_atleast}', df_output.shape[0])
    else:
        df_output = df_output.head(len(df_output))  # choose all

    # filling any empty values from the df_test with dummy values (so that submission doesn't complain)
    # using the df_test
    print('adding', (~df_test['image_path'].isin(df_output['image_path'])).sum(), 'dummy values for test set')

    # only appending rows for "image_path" column that exist in df_test but don't exist in df_output
    df_output = pd.concat([df_output, df_test[~df_test['image_path'].isin(df_output['image_path'])]], ignore_index=True)

    # chosing only smartathon columns
    # df_output['xmin'] = df_output['xmin'].fillna(0).clip(0, 1920)
    # df_output['xmax'] = df_output['xmax'].fillna(0).clip(0, 1920)
    # df_output['ymin'] = df_output['ymin'].fillna(0).clip(0, 1080)
    # df_output['ymax'] = df_output['ymax'].fillna(0).clip(0, 1080)

    if args.setclass is not None:
        df_output['class'] = args.setclass
        df_output['name'] = id2name[args.setclass]

    if args.oversized:
        df_output['xmax'] = -10000
        df_output['ymax'] = -10000
        df_output['xmin'] = 10000
        df_output['ymin'] = 10000
    elif args.assert_valid:
        assert all(df_output['xmin'] >= 0), df_output['xmin'][df_output['xmin'] < 0]
        assert all(df_output['ymin'] >= 0), df_output['ymin'][df_output['ymin'] < 0]
        assert all(df_output['xmax'] <= 1920), df_output['xmax'][df_output['xmax'] > 1920]
        assert all(df_output['ymax'] <= 1080), df_output['ymax'][df_output['ymax'] > 1080]
        assert all(df_output['xmax'] > df_output['xmin'])
        assert all(df_output['ymax'] > df_output['ymin'])
        assert len(df_output) == len(df_test), f'{len(df_output)} == {len(df_test)}'

    outname = 'submission.smartathon'
    if args.setclass:
        outname += f'_allclasses={args.setclass}'
    if args.conf_thresh:
        outname += f'_conf={args.conf_thresh}'
    if args.topk:
        outname += f'_top{args.topk}'
    if args.oversized:
        outname += f'_oversized'
    outname += '.csv'
    outpath = os.path.join(args.labelsdir[0], outname)

    print('total bboxes after all filters:', df_output.shape[0])
    df_output = df_output[['class', 'image_path', 'name', 'xmax', 'xmin', 'ymax', 'ymin']]
    df_output.to_csv(outpath, index=False)
    print('saved to', outpath)
