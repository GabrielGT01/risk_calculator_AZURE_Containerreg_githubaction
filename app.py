
from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app = application 

##Route for Homepage

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predictdata",methods= ['GET','POST'])
def predict_datapoint():
    if request.method =="GET":
        return render_template('home.html')

    else:
        data=CustomData(
        Age=int(request.form.get('Age')),
        AnnualIncome=float(request.form.get('AnnualIncome')),
        CreditScore=float(request.form.get('CreditScore')),
        EmploymentStatus=request.form.get('EmploymentStatus'),
        EducationLevel=request.form.get('EducationLevel'),
        Experience=int(request.form.get('Experience')),
        LoanAmount=float(request.form.get('LoanAmount')),
        LoanDuration=int(request.form.get('LoanDuration')),
        MaritalStatus=request.form.get('MaritalStatus'),
        NumberOfDependents=int(request.form.get('NumberOfDependents')),
        HomeOwnershipStatus=request.form.get('HomeOwnershipStatus'),
        CreditCardUtilizationRate=float(request.form.get('CreditCardUtilizationRate')),
        NumberOfOpenCreditLines=int(request.form.get('NumberOfOpenCreditLines')),
        NumberOfCreditInquiries=int(request.form.get('NumberOfCreditInquiries')),
        BankruptcyHistory=int(request.form.get('BankruptcyHistory')),
        LoanPurpose=request.form.get('LoanPurpose'),
        PreviousLoanDefaults=int(request.form.get('PreviousLoanDefaults')),
        PaymentHistory=float(request.form.get('PaymentHistory')),
        LengthOfCreditHistory=float(request.form.get('LengthOfCreditHistory')),
        SavingsAccountBalance=float(request.form.get('SavingsAccountBalance')),
        CheckingAccountBalance=float(request.form.get('CheckingAccountBalance')),
        TotalAssets=float(request.form.get('TotalAssets')),
        TotalLiabilities=float(request.form.get('TotalLiabilities')),
        UtilityBillsPaymentHistory=float(request.form.get('UtilityBillsPaymentHistory')),
        JobTenure=int(request.form.get('JobTenure')),
        InterestRate=float(request.form.get('InterestRate')),
        MonthlyLoanPayment=float(request.form.get('MonthlyLoanPayment')),
        TotalDebtToIncomeRatio=float(request.form.get('TotalDebtToIncomeRatio'))
        )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline= PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        #print(results)  # Output: array([68.741292]), results[0] is: 68.741292 
        return render_template('home.html',results=results[0])


##if __name__=="__main__":
    #app.run(host="0.0.0.0",debug = True) 
if __name__ == "__main__":
    # Important: Use host="0.0.0.0" to make the app accessible outside the container
    app.run(host="0.0.0.0", debug=True, port=5001)
