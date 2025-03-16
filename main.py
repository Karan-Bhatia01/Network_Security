from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import DataIngestionConfig,DataValidationConfig,DataTransformationConfig
from networksecurity.entity.config_entity import TrainingPipelineConfig
import sys

if __name__=="__main__":
    try:
        trainingpipelineconfig=TrainingPipelineConfig()
        dataingestionconfig=DataIngestionConfig(trainingpipelineconfig)
        data_ingestion=DataIngestion(dataingestionconfig)
        logging.info("Initiate the data ingestion")
        dataingestionartifact=data_ingestion.initiate_data_ingestion()
        logging.info("Data Initation Completed")
        print(dataingestionartifact)
        data_validaiton_config=DataValidationConfig(trainingpipelineconfig)
        data_validation=DataValidation(dataingestionartifact,data_validaiton_config)
        logging.info("Intiate the data validation")
        data_validaiton_artifact=data_validation.intitate_data_validation()
        logging.info("data validation completed")
        print(data_validaiton_artifact)
        data_transformation_config=DataTransformationConfig(trainingpipelineconfig)
        logging.info("data Transformation started")
        data_transformation=DataTransformation(data_validaiton_artifact,data_transformation_config)
        data_transformation_artifact=data_transformation.initiate_data_transformation()
        print(data_transformation_artifact)
        logging.info("data transformation completed")
    except Exception as e:
        raise NetworkSecurityException(e,sys)