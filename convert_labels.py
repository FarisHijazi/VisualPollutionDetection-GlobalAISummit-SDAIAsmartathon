"""
python manip_labels.py repositories/yolov5/runs/detect/exp10/labels/ --rem_conf --topk 2 --output top2
"""
import argparse
import glob
import os

from tqdm import tqdm

if __name__ == '__main__':
    # parser
    parser = argparse.ArgumentParser(__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dir', type=str, help='directory of labels')
    parser.add_argument('--rem_conf', action='store_true', help='remove confidence')
    parser.add_argument('--topk', type=int, default=None, help='choose top k bboxes from confidence')
    parser.add_argument('--conf_thresh', type=float, default=None, help='confidence threshold (before topk)')
    # parser.add_argument('--topk_method', default='conf', choices={'conf', 'size'}, help='set all classes to this value')
    parser.add_argument('--output', type=str, default='output', help='output file')
    args = parser.parse_args()
    args.output = os.path.abspath(args.output)

    root = os.getcwd()
    os.chdir(args.dir)
    os.makedirs(args.output, exist_ok=True)

    label_paths = glob.glob('*.txt')
    # remove classes.txt
    if 'classes.txt' in label_paths:
        label_paths.remove('classes.txt')

    for label_path in tqdm(label_paths):
        with open(label_path, 'r') as f:
            bboxes = [line.strip().split(' ') for line in f.readlines()]
            assert len(bboxes[0]) == 6, 'bbox should have 6 values'

        if args.conf_thresh:
            bboxes = [bbox for bbox in bboxes if float(bbox[-1]) > args.conf_thresh]

        if args.topk:
            bboxes = sorted(bboxes, key=lambda x: float(x[-1]), reverse=True)[: args.topk]

        if args.rem_conf:
            bboxes = [bbox[:-1] for bbox in bboxes]

        with open(os.path.join(args.output, label_path), 'w') as f:
            for bbox in bboxes:
                f.write(' '.join(list(map(str, bbox))) + '\n')

    print('saved results in ', os.path.join(root, args.dir, args.output))
