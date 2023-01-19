# SDAIA Smartathon 2023

```
git clone https://github.com/FarisHijazi/SDAIAsmartathon
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
