from torchvision import transforms


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
