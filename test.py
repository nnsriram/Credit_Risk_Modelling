##########################################################
# The test.py file is used for inference with trained model
##########################################################

import pandas as pd
from sklearn.externals import joblib
from preprocess import preprocess_test

# test_data describes the path of the test file in the form of .csv
test_data = r'C:\Users\Sriram Natarajan\Downloads\ML_Artivatic_dataset\test_indessa.csv'

# the path to the saved model that would be used for prediction
model_file_name = r'F:\models\Log_reg12.sav'
loan_df = pd.read_csv(test_data)

# preprocess the test data - the details of preprocessing are available in preprocess.py
# - input arguements : test data variable
# - output           : preprocessed test data
X_test = preprocess_test(test_data)

# Load model from file
with open(model_file_name, 'rb') as file:
    model = joblib.load(file)

# Generate final predictions
pred = model.predict(X_test)
print(pred)