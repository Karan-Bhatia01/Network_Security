import os
import sys
import numpy as np
import pandas as pd

'''
defining common constant variable for training pipeline
'''
TARGET_COLUMN="Result"
PIPELINE_NAME: str = "NetworkSecurity"
ARTIFACT_DIR: str = "Artifacts"
FILE_NAME: str = "phisingData.csv"

TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

'''
Data Ingestion related constant start with Data_Ingestion var name
'''
DATA_INGESTION_COLLECTION_NAME: str = "NetworkData"
DATA_INGESTION_DATABASE_NAME: str = "AnujBhatia"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_NAME: str = "feature_store"
DATA_INGESTION_INGESTED_NAME: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2