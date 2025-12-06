from datetime import datetime

from src.scripts import load_dataset
from src.services import MachineLearningBuildModelService


class MachineLearningModelGeneratorController:

    @staticmethod
    def generate_and_save():

        dataset = load_dataset()

        start_time = datetime.now()

        model = MachineLearningBuildModelService.train_svm_classifier(dataset)

        MachineLearningBuildModelService.save_model(model, "svm_model.pkl")

        end_time = datetime.now()

        duration = end_time - start_time
        print(
            f"Model training and saving took: {duration.total_seconds() / 60}"
            " minutes"
        )
