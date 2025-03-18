import os
import sys
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.main_utils.utils import save_object, load_object, evaluate_model, load_numpy_array_data
from networksecurity.utils.ml_utils.metric.classification_metric import get_classifications_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def track_mlflow(self,best_model,classificationmetric):
        with mlflow.start_run():
            f1_score=classificationmetric.f1_score
            precision_score=classificationmetric.precision_score
            recall_score=classificationmetric.recall_score

            mlflow.log_metric("f1_score",f1_score)
            mlflow.log_metric("precision",precision_score)
            mlflow.log_metric("recall_score",recall_score)
            mlflow.sklearn.log_model(best_model,"model")



    def train_model(self, X_train, y_train, X_test, y_test):
        models = {
            "Random Forest": RandomForestClassifier(verbose=1),
            "Decision Tree": DecisionTreeClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "Logistic Regression": LogisticRegression(verbose=1),
            "Adaboost": AdaBoostClassifier(),
        }
        params = {
            "Decision Tree": {'criterion': ['gini', 'entropy', 'log_loss']},
            "Random Forest": {'n_estimators': [8, 16, 32, 64, 69, 128, 256]},
            "Gradient Boosting": {
                'learning_rate': [0.1, 0.01, 0.05, 0.001, 0.314],
                #'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
               # 'n_estimators': [8, 16, 32, 64, 128, 256]
            },
            "Logistic Regression": {},
            "AdaBoost": {
                'learning_rate': [0.1, 0.01, 0.05, 0.001, 0.314],
                'n_estimators': [8, 16, 32, 64, 128, 256]
            }
        }

        model_report = evaluate_model(X_train, y_train, X_test, y_test, models, param=params)
        best_model_name = max(model_report, key=model_report.get)
        best_model = models[best_model_name]

        y_train_pred = best_model.predict(X_train)
        classification_train_metric = get_classifications_score(y_train, y_train_pred)
        # Track Mlflow

        self.track_mlflow(best_model,classification_train_metric)


        y_test_pred = best_model.predict(X_test)
        classification_test_metric = get_classifications_score(y_test, y_test_pred)
        self.track_mlflow(best_model,classification_test_metric)
        preprocessor = load_object(self.data_transformation_artifact.transformed_object_file_path)
        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path, exist_ok=True)

        network_model = NetworkModel(preprocessor=preprocessor, model=best_model)
        save_object(self.model_trainer_config.trained_model_file_path, obj=network_model)

        return ModelTrainerArtifact(
            trained_model_file_path=self.model_trainer_config.trained_model_file_path,
            train_metric_artifact=classification_train_metric,
            test_metric_artifact=classification_test_metric
        )

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            x_train, y_train, x_test, y_test = train_arr[:, :-1], train_arr[:, -1], test_arr[:, :-1], test_arr[:, -1]
            y_train = y_train.ravel()
            y_test = y_test.ravel()

            return self.train_model(x_train, y_train, x_test, y_test)

        except Exception as e:
            raise NetworkSecurityException(e, sys)
