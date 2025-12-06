from os import makedirs
from os.path import abspath, basename, exists, join
from shutil import copyfile

import numpy as np

from .load_dataset import load_dataset


def build_train_dataset(
    train_size: float = 0.8,
    test_size: float = 0.1,
    val_size: float = 0.1,
    output_dataset: str = "dataset"
) -> None:
    """
    Build and save the training, testing, and validation datasets.
    The datasets are split according to the specified sizes.

    Args:
        train_size (float): Proportion of the dataset to include in the
            training set.
        test_size (float): Proportion of the dataset to include in the
            testing set.
        val_size (float): Proportion of the dataset to include in the
            validation set.
        output_dataset (str): The directory to save the datasets.
    Returns:
        None
    """

    dataset = load_dataset()

    print("Building dataset...")

    if not exists(output_dataset):
        makedirs(output_dataset, exist_ok=True)

    for key, images in dataset.items():
        np.random.shuffle(images)

        total_images = len(images)

        train_end = int(total_images * train_size)
        test_end = train_end + int(total_images * test_size)

        train_set = images[:train_end]
        test_set = images[train_end:test_end]

        val_set = images[test_end:]

        train_dir = join(output_dataset, "train", key)
        test_dir = join(output_dataset, "test", key)
        val_dir = join(output_dataset, "val", key)

        makedirs(train_dir, exist_ok=True)
        makedirs(test_dir, exist_ok=True)
        makedirs(val_dir, exist_ok=True)

        for img_path in train_set:
            dest_path = join(train_dir, basename(img_path))
            copyfile(abspath(img_path), dest_path)

        print(
            "Training set for class",
            key,
            "built with",
            len(train_set),
            "images."
        )

        for img_path in test_set:
            dest_path = join(test_dir, basename(img_path))
            copyfile(abspath(img_path), dest_path)

        print(
            "Testing set for class",
            key,
            "built with",
            len(test_set),
            "images."
        )

        for img_path in val_set:
            dest_path = join(val_dir, basename(img_path))
            copyfile(abspath(img_path), dest_path)

        print(
            "Validation set for class",
            key,
            "built with",
            len(val_set),
            "images."
        )
    print("Dataset building complete.")