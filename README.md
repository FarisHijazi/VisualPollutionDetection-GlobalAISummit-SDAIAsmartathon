# SDAIA Smartathon 2023

- theefaris@gmail.com
- serrymrss@gmail.com
- ibraheemkhurfan@gmail.com
- Shyom.1417@gmail.com
- rashamalqahtani@gmail.com

We used a combination of transfer learning and fine-tuning to train our model.
We did a lot of data preprocessing and model tuning and ensemble to get the best results and use black box testing to get the best results.

We used the following models:

YoloV5x and yolov5x6

There is much more experiment and trianing code that wasn't included, but everything needed to run the model is included in this repo.

## Inference

to try out a few images, use the gradio demo:

```sh
python demo.py
```

To run inference on a folder of images, use the following command (you may have to run this many times to reproduce the results):

```sh
git clone https://github.com/FarisHijazi/SDAIAsmartathon
mkdir repositories/
git clone https://github.com/ultralytics/yolov5 repositories/yolov5


# run prediction on 3 models
# note that --augment will produce different results each time, so you may have to run this many times to reproduce the results
cd repositories/yolov5
python detect.py --weights ../../yolov5s/weights/best.pt ../../exp88/weights/best.pt ../../yolov5x6-heavyaug-mergetesttrain2/weights/best.pt --augment --source path/to/data

# convert predictions from txt files to csv files
# (change exp1234 to the experiment number you want to run)
cd ../..
python cocodir2submission.py repositories/yolov5/runs/detect/exp1234/labels/ --conf_thresh .42
```

## setup minio for DVC

```sh
docker run --name Minio -e MINIO_ROOT_USER=admin -e MINIO_ROOT_PASSWORD=supersecret -p 9000:9000 -p 9001:9001 -v /d/data/minio:/data -d quay.io/minio/minio:latest server /data --console-address ":9001"
```

## Repos

- https://github.com/sekilab/RoadDamageDetector
- https://colab.research.google.com/drive/1X9A8odmK4k6l26NDviiT6dd6TgR-piOa

## class distribution of submissions

| all this class | class name | score |
| --- | --- | --- |
| 0 | GRAFFITI | 6.86638 |
| 1 | FADED_SIGNAGE | 0.95170 |
| 2 | POTHOLES | 10.68917 |
| 3 | GARBAGE | 36.69862 |
| 4 | CONSTRUCTION_ROAD | 9.37099 |
| 5 | BROKEN_SIGNAGE | 0.48359 |
| 6 | BAD_STREETLIGHT | 0.03187 |
| 7 | BAD_BILLBOARD | 10.49133 |
| 8 | SAND_ON_ROAD | 4.95140 |
| 9 | CLUTTER_SIDEWALK | 10.45390 |
| 10 | UNKEPT_FACADE | 0.66649 |

summation: 91.65544000000001
