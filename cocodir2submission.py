"""
convert yolo format to smartathon stupid format
read coco dir txt files and pandas dataframe and puts them in a csv file ready to submit

repositories/yolov5/runs/detect/exp12/labels
"""
import argparse
import glob
import multiprocessing as mp
import os

import pandas as pd
from tqdm import tqdm

import bbox


def cocodir2df(globstr='train/labels/*.txt'):
    label_paths = glob.glob(globstr)
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
                conf = 0

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
    parser.add_argument(
        'labelsdir', default='repositories/yolov5/runs/detect/exp12/labels', help='glob string for txt files'
    )
    parser.add_argument('-d', '--data', default='data/Challenge1/', help='data dir')
    overrides = parser.add_argument_group('overrides')
    overrides.add_argument('--setclass', type=int, default=None, help='set all classes to this value')
    overrides.add_argument('--oversized', action='store_true', help='set all classes to this value')
    args = parser.parse_args()

    df = pd.read_csv(args.data + '/train.csv')
    id2name = dict(list(zip(df['class'], df['name'])))

    df_test = pd.read_csv(args.data + '/test.csv')
    df_test['class'] = 3
    df_test['name'] = 'GARBAGE'
    df_test['xmin'] = -10000
    df_test['ymin'] = -10000
    df_test['xmax'] = 10000
    df_test['ymax'] = 10000
    df_test = df_test[['class', 'image_path', 'name', 'xmax', 'xmin', 'ymax', 'ymin']]

    df_output = cocodir2df(args.labelsdir + '/*.txt')
    df_output['name'] = df_output['class'].apply(lambda x: id2name[int(x)])

    ## convert coco format to smartathon format
    df_output[['xmin', 'ymin', 'xmax', 'ymax']] = (
        df_output[['x', 'y', 'w', 'h']].astype(float).apply(bbox.coco2smartathon, axis=1, result_type='expand')
    )

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
        df_output['xmax'] = -100000
        df_output['ymax'] = -100000
        df_output['xmin'] = 100000
        df_output['ymin'] = 100000
    else:
        assert all(df_output['xmin'] >= 0), df_output['xmin']
        assert all(df_output['ymin'] >= 0), df_output['ymin']
        assert all(df_output['xmax'] <= 1920), df_output['xmax']
        assert all(df_output['ymax'] <= 1080), df_output['ymax']
        assert all(df_output['xmax'] > df_output['xmin'])
        assert all(df_output['ymax'] > df_output['ymin'])
        assert len(df_output) == len(df_test), f'{len(df_output)} == {len(df_test)}'

    # groupby image_path and sort by "conf" and choose best confidence row
    df_output = df_output.sort_values(by=['image_path', 'conf'], ascending=False).groupby('image_path').head(1)
    # outpath = os.path.join(args.labelsdir, 'submission.smartathon.csv')
    outpath = os.path.join(
        args.labelsdir, 'submission.smartathon_cls={setclass}_oversized={oversized}.csv'.format(**vars(args))
    )

    df_output[['class', 'image_path', 'name', 'xmax', 'xmin', 'ymax', 'ymin']].to_csv(outpath, index=False)
    print('saved to', outpath)
