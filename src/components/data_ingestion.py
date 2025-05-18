
import os
import sys
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig


from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer


# This creates a simple class that stores file paths:
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")
    source_data_path: str = "loan.csv"

class DataIngestion:
    def __init__(self):
        # ingestion_config would save the four variables from class ingestionconfig
        self.ingestion_config = DataIngestionConfig()
        

    def start_data_ingestion(self):
        """Process data ingestion from source CSV to train/test splits
        
        Returns:
            Tuple containing paths to training and testing data files
            
        Raises:
            Exception: If any error occurs during ingestion process
        """
        # this reads the dataset, can be local or from a database
        logging.info("Entered the data ingestion method or components")
        try:
            # Check if source file exists
            if not os.path.exists(self.ingestion_config.source_data_path):
                error_msg = f"Source data file not found: {self.ingestion_config.source_data_path}"
                logging.error(error_msg)
                raise FileNotFoundError(error_msg)
                
            # Read the dataset
            df = pd.read_csv(self.ingestion_config.source_data_path)
            logging.info(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
            
            ##columns to be dropped, i already worked on the data and know columns that are not neede
            drop_cols = ['ApplicationDate', 'MonthlyIncome', 'BaseInterestRate', 'NetWorth', 'LoanApproved','DebtToIncomeRatio','MonthlyDebtPayments' ]
            df = df.drop(columns=drop_cols)
            print("Remaining columns after drop:", df.columns.tolist())

            
            # create directory to store all the files
            # Creates the "artifacts" directory if it doesn't already exist
            # remember self.ingestion_config.train_data_path is os.path.join('artifacts',"train.csv")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info('Train test split initiated')
            logging.info("Performing train-test split")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            
            # initiate and train to these folders
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            logging.info(f"Saved training data with {train_set.shape[0]} rows to {self.ingestion_config.train_data_path}")

            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info(f"Saved testing data with {test_set.shape[0]} rows to {self.ingestion_config.test_data_path}")

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as e:
            # this prints the error and re-raises it so it's not silently ignored
            logging.error(f"Error during data ingestion: {e}")
            raise e


if __name__ == "__main__":
    try:
        obj = DataIngestion()
        train_data, test_data = obj.start_data_ingestion()
        print("Data ingestion completed successfully!")
        print(f"Training data: {train_data}")
        print(f"Testing data: {test_data}")
        
        data_transformation = DataTransformation()
        train_arr,test_arr,transformer_path = data_transformation.initiate_data_transformation(train_data, test_data)
        print('the pickle file for transformer has been created')

        modeltrainer = ModelTrainer()
        modeltrainer.initiate_model_trainer(train_arr,test_arr)
        print("model sucessfully trained")
        
        
        
        
    except Exception as err:
        print(f"Failed to ingest data or transform data: {err}")
