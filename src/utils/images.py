import cv2
import numpy as np
from skimage.feature import hog


from glob import glob


def get_image_files(
    directory: str,
    extensions: list[str] = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.gif'],
    exclude_labels: list[str] = [],
):
    """
    Retrieve a list of image files from the specified directory with given
    extensions.

    Args:
        directory (str): The directory to search for image files.
        extensions (list): A list of file extensions to look for.
        exclude_labels (list): A list of labels; files containing these labels
            in their names will be excluded.
    Returns:
        list: A list of image file paths.
    """
    image_files = []
    for ext in extensions:
        image_files.extend(glob(f"{directory}/{ext}"))

    if exclude_labels:
        image_files = [
            img for img in image_files
            if not any(label in img for label in exclude_labels)
        ]

    return image_files


def preprocess_image(path: str) -> np.ndarray:
    """
    Preprocess an image by reading it, converting to grayscale, resizing,
    and applying Gaussian blur.

    Args:
        path (str): The file path to the image.
    Returns:
        np.ndarray: The preprocessed image.
    """
    img = cv2.imread(path)
    img = cv2.imread(path)
    img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)  # type: ignore
    img = cv2.resize(img, (256, 256))
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img


def extract_hog_features(gray: np.ndarray) -> np.ndarray:
    """
    Extract HOG (Histogram of Oriented Gradients) features from a grayscale
    image.

    Args:
        gray (np.ndarray): The grayscale image.
    Returns:
        np.ndarray: The HOG feature vector.
    """
    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm='L2-Hys'
    )
    return features


def extract_features(path: str) -> np.ndarray:
    """
    Extract HOG features from an image at the specified path.

    Args:
        path (str): The file path to the image.
    Returns:
        np.ndarray: The extracted HOG feature vector.
    """
    gray = preprocess_image(path)
    hog_feat = extract_hog_features(gray)
    return hog_feat
