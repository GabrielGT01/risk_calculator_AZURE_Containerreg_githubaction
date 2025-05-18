


import sys
from src.logger import logging
import os
from dataclasses import dataclass

import numpy as np 
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_transfomer_path: str = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        this function will transform the data
        """
        try:
            ##define all variables needed for the preprocessor
            numeric_features = ['Age','AnnualIncome','CreditScore','Experience',
                                'LoanAmount','LoanDuration','NumberOfDependents','CreditCardUtilizationRate',
                                 'NumberOfOpenCreditLines','NumberOfCreditInquiries','BankruptcyHistory','PreviousLoanDefaults',
                                 'PaymentHistory','LengthOfCreditHistory','SavingsAccountBalance','CheckingAccountBalance',
                                 'TotalAssets','TotalLiabilities','UtilityBillsPaymentHistory','JobTenure',
                                 'InterestRate','MonthlyLoanPayment','TotalDebtToIncomeRatio']
            
            categorical_features = ['HomeOwnershipStatus','MaritalStatus', 'LoanPurpose']
            education_feature = ['EducationLevel']
            employment_feature = ['EmploymentStatus']
            
            # Custom orders to avoid dcurse of dimensionality
            ##target encoding could also be done to other categorical features
            edu_order = [['High School', 'Associate', 'Bachelor', 'Master', 'Doctorate']]
            emp_order = [['Unemployed', 'Employed', 'Self-Employed']]

            # Define transformers / pipeline
            num_pipeline = Pipeline(
                steps= [
                    ('imputer', SimpleImputer(strategy = "median")),
                    ('scaler', StandardScaler()), 
                    ]
            )
            cat_pipeline = Pipeline(
                steps = [
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder(handle_unknown="ignore",drop='first'))
                ]
            )
            education_pipeline = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('ordinal_encoder', OrdinalEncoder(categories=edu_order))
                ])

            employment_pipeline = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('ordinal_encoder', OrdinalEncoder(categories=emp_order))
                ])
            
            # show me what features are being used
            logging.info(f"Categorical columns: {categorical_features + education_feature + employment_feature}")
            logging.info(f"Numerical columns: {numeric_features}")

            # Create column transformer with all transformers
            # Combine all

            logging.info("creating the joint preprocessor")
            
            preprocessor = ColumnTransformer(transformers=[
                ('num', num_pipeline, numeric_features),
                ('nominal_cat', cat_pipeline, categorical_features),
                ('education_ord', education_pipeline, education_feature),
                ('employment_ord', employment_pipeline, employment_feature)
            ])

            logging.info("the preprocessor works")
            return preprocessor
        
        except Exception as e:
            # This catches and logs the error so we know what failed
            logging.error(f"Error in get_data_transformer_object: {e}")
            raise e

    def initiate_data_transformation(self, train_path, test_path):
        """
        Aim is to transform the data from data ingestion
        """

        try:
            # load the csv file
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            # get the transformer pkl
            preprocessing_transformer = self.get_data_transformer_object()

            numeric_features = ['Age','AnnualIncome','CreditScore','Experience',
                                'LoanAmount','LoanDuration','NumberOfDependents','CreditCardUtilizationRate',
                                 'NumberOfOpenCreditLines','NumberOfCreditInquiries','BankruptcyHistory','PreviousLoanDefaults',
                                 'PaymentHistory','LengthOfCreditHistory','SavingsAccountBalance','CheckingAccountBalance',
                                 'TotalAssets','TotalLiabilities','UtilityBillsPaymentHistory','JobTenure',
                                 'InterestRate','MonthlyLoanPayment','TotalDebtToIncomeRatio']
            
            categorical_features = ['HomeOwnershipStatus','MaritalStatus', 'LoanPurpose']
            education_feature = ['EducationLevel']
            employment_feature = ['EmploymentStatus']
            

            # define the column with the target
            target_column_name = "RiskScore"

            # separate X and y
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )
            expected_columns = numeric_features + categorical_features + education_feature + employment_feature
            missing_columns = [col for col in expected_columns if col not in input_feature_train_df.columns]

            if missing_columns:
                raise ValueError(f"The following expected columns are missing from the dataset: {missing_columns}")

            # transform X
            input_feature_train_arr = preprocessing_transformer.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_transformer.transform(input_feature_test_df)

            ### column-wise concatenation of X, y 
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            # save the preprocessor pkl file
            save_object(
                file_path=self.data_transformation_config.preprocessor_transfomer_path,
                obj=preprocessing_transformer
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_transfomer_path,
            )
        except Exception as e:
            # This logs and rethrows the exception to avoid silent failure
            logging.error(f"Error in initiate_data_transformation: {e}")
            raise e

        
           
