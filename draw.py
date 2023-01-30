"""
Draws bounding boxes on images and saves them to drawn/ folder.
"""
import argparse
import os
from multiprocessing.pool import ThreadPool

import cv2
import pandas as pd
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data', default='data/Challenge1/', help='glob string for txt files')
    parser.add_argument('-t', '--traincsv', default='train.csv', help='train csv file')
    parser.add_argument('-o', '--output', default='{data}/drawn', help='output dir')
    args = parser.parse_args()

    args.output = args.output.format(data=args.data)

    os.chdir(args.data)

    os.makedirs(args.output, exist_ok=True)
    df = pd.read_csv(args.traincsv)

    id2name = dict(list(zip(df['class'], df['name'])))
    # create random colors for each class in rgb format as tuples that will feed to cv2.rectangle
    # id2color = {k: tuple(np.random.randint(0, 255, 3).tolist()) for k in id2name.keys()}
    id2color = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (125, 0, 0),
        (0, 125, 0),
        (0, 0, 125),
        (125, 125, 0),
        (125, 0, 125),
        (0, 125, 125),
        (255, 125, 0),
        (255, 0, 125),
        (0, 255, 125),
        (125, 255, 0),
        (125, 0, 255),
        (0, 125, 255),
        (255, 255, 125),
        (255, 125, 255),
        (125, 255, 255),
        (255, 125, 125),
        (125, 255, 125),
        (125, 125, 255),
    ]

    def draw(impath_subdf):
        impath, subdf = impath_subdf
        impath = 'images/' + impath
        im = cv2.imread(impath)
        with open(impath.replace('/images/', '/labels/').replace('.jpg', '.txt'), 'w') as f:
            for _, [cls, image_path, name, xmax, xmin, ymax, ymin] in subdf.iterrows():
                x1, y1, x2, y2 = xmin * 2, ymin * 2, xmax * 2, ymax * 2
                im = cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), id2color[int(cls)], 2)
                im = cv2.putText(
                    im,
                    id2name[cls],
                    (int(x1), int(y1)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    id2color[int(cls)],
                    2,
                    cv2.LINE_AA,
                )

        cv2.imwrite(args.output + '/' + os.path.split(impath)[-1], im)

    groups = df.groupby('image_path')
    # for impath, subdf in tqdm(groups):
    #     draw(impath, subdf)

    list(tqdm(ThreadPool(8).imap_unordered(draw, groups), total=len(groups)))
