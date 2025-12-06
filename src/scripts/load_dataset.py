from src.core.constants import (
    CPF_IMAGE_FOLDERS,
    CNH_IMAGE_FOLDERS,
    RG_IMAGE_FOLDERS
)
from src.utils import get_image_files


def load_dataset() -> dict[str, list[str]]:
    """
    Load the dataset of images from predefined folders.
    Returns a dictionary with keys 'cpf', 'cnh', and 'rg' mapping to lists of
    image file paths.
    """

    cpf = []

    for folder in CPF_IMAGE_FOLDERS:
        images = get_image_files(folder, exclude_labels=["segmentation"])
        cpf.extend(images)

    cnh = []

    for folder in CNH_IMAGE_FOLDERS:
        images = get_image_files(folder, exclude_labels=["segmentation"])
        cnh.extend(images)

    rg = []

    for folder in RG_IMAGE_FOLDERS:
        images = get_image_files(folder, exclude_labels=["segmentation"])
        rg.extend(images)

    dataset = {
        "cpf": cpf,
        "cnh": cnh,
        "rg": rg
    }

    return dataset
