import cv2
import numpy as np


def annoatet_mask_gray(_image_gray, _mask, _color, _rate):
    annoatte_image = _image_gray.copy()
    to_annotate_mask = ((_mask > 0) * 255).astype(np.uint8)
    np.putmask(annoatte_image, to_annotate_mask, _color)
    annoatte_image = _rate * annoatte_image + (1 - _rate) * _image_gray
    return annoatte_image


def annoatate_mask_bgr(_image, _mask, _color, _rate):
    annotate_channels = []
    for channel_image, channel_color in zip(cv2.split(_image), _color):
        annotate_channels.append(annoatet_mask_gray(channel_image, _mask, channel_color, _rate))
    annotate_image = np.stack(annotate_channels, axis=2)
    return annotate_image


def mask2box(_mask):
    _, _, stats, _ = cv2.connectedComponentsWithStats(_mask)
    stats = stats[stats[:, 4].argsort()]
    return stats[:-1]
