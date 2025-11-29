import pickle

from sklearn.svm import LinearSVC

from src.utils import extract_features


class MachineLearningBuildModelService:

    @staticmethod
    def train_svm_classifier(
        dataset: dict[str, list[str]],
    ) -> LinearSVC:
        """
        Trains a Linear SVM classifier on the provided dataset.

        Args:
            dataset (dict[str, list[str]]): A dictionary where keys are class
                names and values are lists of image file paths.

        Returns:
            LinearSVC: The trained Linear SVM classifier.
        """
        X, y = [], []
        classes = list(dataset.keys())

        for class_name in classes:
            for image in dataset[class_name]:
                feature = extract_features(image)
                X.append(feature)
                y.append(class_name)

        model = LinearSVC()
        model.fit(X, y)
        return model

    @staticmethod
    def save_model(model: LinearSVC, file_path: str) -> None:
        """
        Saves the trained model to a file.

        Args:
            model (LinearSVC): The trained Linear SVM classifier.
            file_path (str): The file path where the model should be saved.
        """
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)

    @staticmethod
    def load_model(file_path: str) -> LinearSVC:
        """
        Loads a trained model from a file.

        Args:
            file_path (str): The file path from which to load the model.
        """
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        return model


class MachineLearningPredictService:

    def __init__(self, model: LinearSVC) -> None:
        self.model = model

    def predict_image_class(
        self,
        image_path: str,
    ) -> str:
        """
        Predicts the class of a given image using the trained model.

        Args:
            model (LinearSVC): The trained Linear SVM classifier.
            image_path (str): The file path of the image to be classified.

        Returns:
            str: The predicted class name of the image.
        """
        feature = extract_features(image_path)
        prediction = self.model.predict([feature])
        return prediction[0]
