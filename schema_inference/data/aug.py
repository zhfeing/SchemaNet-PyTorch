from typing import Dict
import torchvision.transforms.functional as TF

import cv_lib.augmentation as aug


mnist_train_aug = aug.Compose(
    aug.RandomRotation((-30, 30))
)

cifar_train_aug = aug.Compose(
    aug.RandomCrop((32, 32), padding=4),
    aug.RandomHorizontalFlip()
)

imagenet_train_aug = aug.Compose(
    aug.RandomResizedCrop(size=(224, 224), scale=(0.6, 1)),
    aug.RandomHorizontalFlip()
)
imagenet_val_aug = aug.Compose(
    aug.Resize(256, mode=TF.InterpolationMode.BICUBIC),
    aug.CenterCrop((224, 224))
)


__REGISTERED_AUG__: Dict[str, aug.Compose] = {
    "mnist_train": mnist_train_aug,
    "mnist_val": None,
    "cifar_10_train": cifar_train_aug,
    "cifar_10_val": None,
    "cifar_100_train": cifar_train_aug,
    "cifar_100_val": None,
    "imagenet_train": imagenet_train_aug,
    "imagenet_val": imagenet_val_aug,
    "imagenet=10_train": imagenet_train_aug,
    "imagenet=10_val": imagenet_val_aug,
    "imagenet=10birds_train": imagenet_train_aug,
    "imagenet=10birds_val": imagenet_val_aug,
    "imagenet=20_train": imagenet_train_aug,
    "imagenet=20_val": imagenet_val_aug,
    "imagenet=50_train": imagenet_train_aug,
    "imagenet=50_val": imagenet_val_aug,
    "imagenet=100_train": imagenet_train_aug,
    "imagenet=100_val": imagenet_val_aug,
    "cub_200_train": imagenet_train_aug,
    "cub_200_val": imagenet_val_aug,
    "stanford_cars_train": imagenet_train_aug,
    "stanford_cars_val": imagenet_val_aug,
    "prob_dataset_train": imagenet_train_aug,
    "prob_dataset_val": imagenet_val_aug,
    "caltech_101_train": imagenet_train_aug,
    "caltech_101_val": imagenet_val_aug,
    "caltech_101_1_train": imagenet_train_aug,
    "caltech_101_1_val": imagenet_val_aug,
    "mini_imagenet_train": imagenet_train_aug,
    "mini_imagenet_val": imagenet_val_aug,
    "imagenet_a_val": imagenet_val_aug,
    "imagenet_r_val": imagenet_val_aug
}


def get_data_aug(dataset_name: str, split: str):
    if "mnist" in dataset_name.lower():
        dataset_name = "mnist"
    name = "{}_{}".format(dataset_name, split)
    return __REGISTERED_AUG__[name]

