##WARNING: don't forget that smartathon format actually has (xmax, xmin, ymax, ymin)
def xyxy2smartathon(xyxy):
    x1, y1, x2, y2 = xyxy
    return [x1 * 1920 / 2, y1 * 1080 / 2, x2 * 1920 / 2, y2 * 1080 / 2]


def smartathon2xyxy(xyxy):
    x1, y1, x2, y2 = xyxy
    x1, y1, x2, y2 = [x1 * 2 / 1920, y1 * 2 / 1080, x2 * 2 / 1920, y2 * 2 / 1080]
    return [x1, y1, x2, y2]
    # return list(map(lambda x: np.clip(x, 0, 1), [x1, y1, x2, y2]))


def xyxy2xywh(xyxy):
    x1, y1, x2, y2 = xyxy
    h = y2 - y1
    w = x2 - x1
    result = [x1 + w / 2, y1 + h / 2, w, h]
    return result
    # return list(map(lambda x: np.clip(x, 0, 1), result))


def xywh2xyxy(xywh):
    x, y, w, h = xywh
    return [x - w / 2, y - h / 2, x + w / 2, y + h / 2]


def smartathon2coco(xyxy):
    x1, y1, x2, y2 = xyxy
    # smartathon2xyxy
    x1, y1, x2, y2 = [x1 * 2 / 1920, y1 * 2 / 1080, x2 * 2 / 1920, y2 * 2 / 1080]

    h = y2 - y1
    w = x2 - x1
    xywh = [x1 + w / 2, y1 + h / 2, w, h]
    return xywh
    # return xyxy2xywh(smartathon2xyxy([x1, y1, x2, y2]))


def coco2smartathon(xywh):
    x, y, w, h = xywh
    # xywh2xyxy
    x1, y1, x2, y2 = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]
    # xyxy2smartathon
    x1, y1, x2, y2 = [x1 * 1920 / 2, y1 * 1080 / 2, x2 * 1920 / 2, y2 * 1080 / 2]
    return [x1, y1, x2, y2]


if __name__ == '__main__':
    _1234 = [1.0, 2.0, 3.0, 4.0]
    assert xyxy2smartathon(smartathon2xyxy(_1234)) == _1234, f'{xyxy2smartathon(smartathon2xyxy(_1234))} != {_1234}'
    assert smartathon2xyxy(xyxy2smartathon(_1234)) == _1234, f'{smartathon2xyxy(xyxy2smartathon(_1234))} != {_1234}'
    assert xyxy2xywh(xywh2xyxy(_1234)) == _1234, f'{xyxy2xywh(xywh2xyxy(_1234))} != {_1234}'
    assert xywh2xyxy(xyxy2xywh(_1234)) == _1234, f'{xywh2xyxy(xyxy2xywh(_1234))} != {_1234}'
    assert coco2smartathon(smartathon2coco(_1234)) == _1234, f'{coco2smartathon(smartathon2coco(_1234))} != {_1234}'
    assert smartathon2coco(coco2smartathon(_1234)) == _1234, f'{smartathon2coco(coco2smartathon(_1234))} != {_1234}'
