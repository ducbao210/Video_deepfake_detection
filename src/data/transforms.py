import random

import cv2
import numpy as np
from PIL import Image
from torchvision import transforms


class RandomJPEGCompression:
    """
    Recompress an image using a randomly selected JPEG quality in the range [qmin, qmax].
    Place this transform after ToPILImage and before ToTensor.
    """

    def __init__(self, qmin=60, qmax=100, p=0.5):
        self.qmin = int(qmin)
        self.qmax = int(qmax)
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img

        quality = random.randint(self.qmin, self.qmax)
        arr = np.array(img)[:, :, ::-1]  # RGB -> BGR cho cv2

        ok, buf = cv2.imencode(".jpg", arr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return img

        decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return Image.fromarray(decoded[:, :, ::-1])  # BGR -> RGB

    def __repr__(self):
        return f"{self.__class__.__name__}(q=[{self.qmin},{self.qmax}], p={self.p})"


def get_train_transforms(cfg):
    image_size = cfg.dataset.image_size

    aug = cfg.transforms.augmentation
    norm = cfg.transforms.normalization

    transform_list = [
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
    ]

    if aug.apply:
        transform_list.extend(
            [
                transforms.RandomHorizontalFlip(p=aug.horizontal_flip_prob),
                transforms.ColorJitter(
                    brightness=aug.brightness,
                    contrast=aug.contrast,
                    saturation=aug.saturation,
                    hue=aug.hue,
                ),
            ]
        )

        jpeg_cfg = aug.get("jpeg_quality", None)
        if jpeg_cfg is not None:
            transform_list.append(
                RandomJPEGCompression(
                    qmin=jpeg_cfg.min,
                    qmax=jpeg_cfg.max,
                    p=jpeg_cfg.get("prob", 0.5),
                )
            )

    transform_list.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=norm.mean,
                std=norm.std,
            ),
        ]
    )

    return transforms.Compose(transform_list)


def get_val_transforms(cfg):
    image_size = cfg.dataset.image_size
    norm = cfg.transforms.normalization

    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=norm.mean,
                std=norm.std,
            ),
        ]
    )
