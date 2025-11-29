from datetime import datetime

from src.core.constants import (
    CPF_IMAGE_FOLDERS,
    CNH_IMAGE_FOLDERS,
    RG_IMAGE_FOLDERS
)
from src.services import MachineLearningBuildModelService
from src.utils import get_image_files


class MachineLearningModelGeneratorController:

    @staticmethod
    def generate_and_save():

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

        start_time = datetime.now()

        model = MachineLearningBuildModelService.train_svm_classifier(dataset)

        MachineLearningBuildModelService.save_model(model, "svm_model.pkl")

        end_time = datetime.now()

        duration = end_time - start_time
        print(
            f"Model training and saving took: {duration.total_seconds() / 60}"
            " minutes"
        )
