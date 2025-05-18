
import sys
import pandas as pd
import numpy as np
from src.logger import logging

from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            model_path ="artifacts\model.pkl"
            preprocessor_path = "artifacts\preprocessor.pkl"
            model = load_object(file_path = model_path)
            preprocessor= load_object(file_path = preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise(f"Error arising from loading the transformers {e}")
        
        
        

class CustomData:
    def __init__(self, Age, AnnualIncome, CreditScore, EmploymentStatus,
                 EducationLevel, Experience, LoanAmount, LoanDuration,
                 MaritalStatus, NumberOfDependents, HomeOwnershipStatus,
                 CreditCardUtilizationRate, NumberOfOpenCreditLines, NumberOfCreditInquiries,
                 BankruptcyHistory, LoanPurpose, PreviousLoanDefaults, PaymentHistory, LengthOfCreditHistory,
                 SavingsAccountBalance, CheckingAccountBalance, TotalAssets,
                 TotalLiabilities, UtilityBillsPaymentHistory, JobTenure,
                 InterestRate, MonthlyLoanPayment, TotalDebtToIncomeRatio):

        self.Age = Age
        self.AnnualIncome = AnnualIncome
        self.CreditScore = CreditScore
        self.EmploymentStatus = EmploymentStatus
        self.EducationLevel = EducationLevel
        self.Experience = Experience
        self.LoanAmount = LoanAmount
        self.LoanDuration = LoanDuration
        self.MaritalStatus = MaritalStatus
        self.NumberOfDependents = NumberOfDependents
        self.HomeOwnershipStatus = HomeOwnershipStatus
        self.CreditCardUtilizationRate = CreditCardUtilizationRate
        self.NumberOfOpenCreditLines = NumberOfOpenCreditLines
        self.NumberOfCreditInquiries = NumberOfCreditInquiries
        self.BankruptcyHistory = BankruptcyHistory
        self.LoanPurpose = LoanPurpose
        self.PreviousLoanDefaults = PreviousLoanDefaults
        self.PaymentHistory = PaymentHistory
        self.LengthOfCreditHistory = LengthOfCreditHistory
        self.SavingsAccountBalance = SavingsAccountBalance
        self.CheckingAccountBalance = CheckingAccountBalance
        self.TotalAssets = TotalAssets
        self.TotalLiabilities = TotalLiabilities
        self.UtilityBillsPaymentHistory = UtilityBillsPaymentHistory
        self.JobTenure = JobTenure
        self.InterestRate = InterestRate
        self.MonthlyLoanPayment = MonthlyLoanPayment
        self.TotalDebtToIncomeRatio = TotalDebtToIncomeRatio

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Age": [self.Age],
                "AnnualIncome": [self.AnnualIncome],
                "CreditScore": [self.CreditScore],
                "EmploymentStatus": [self.EmploymentStatus],
                "EducationLevel": [self.EducationLevel],
                "Experience": [self.Experience],
                "LoanAmount": [self.LoanAmount],
                "LoanDuration": [self.LoanDuration],
                "MaritalStatus": [self.MaritalStatus],
                "NumberOfDependents": [self.NumberOfDependents],
                "HomeOwnershipStatus": [self.HomeOwnershipStatus],
                "CreditCardUtilizationRate": [self.CreditCardUtilizationRate],
                "NumberOfOpenCreditLines": [self.NumberOfOpenCreditLines],
                "NumberOfCreditInquiries": [self.NumberOfCreditInquiries],
                "BankruptcyHistory": [self.BankruptcyHistory],
                "LoanPurpose": [self.LoanPurpose],
                "PreviousLoanDefaults": [self.PreviousLoanDefaults],
                "PaymentHistory": [self.PaymentHistory],
                "LengthOfCreditHistory": [self.LengthOfCreditHistory],
                "SavingsAccountBalance": [self.SavingsAccountBalance],
                "CheckingAccountBalance": [self.CheckingAccountBalance],
                "TotalAssets": [self.TotalAssets],
                "TotalLiabilities": [self.TotalLiabilities],
                "UtilityBillsPaymentHistory": [self.UtilityBillsPaymentHistory],
                "JobTenure": [self.JobTenure],
                "InterestRate": [self.InterestRate],
                "MonthlyLoanPayment": [self.MonthlyLoanPayment],
                "TotalDebtToIncomeRatio": [self.TotalDebtToIncomeRatio]
            }

            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise Exception(f"Error creating DataFrame from input data: {e}")
