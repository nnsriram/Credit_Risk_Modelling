##########################################################
# The preprocess.py file is used for preprocessing the data which includes
# - cleaning the data
# - data transformations and scaling
##########################################################


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

# preprocess_split_train: includes basic preprocessing operations on the training
# data and generates train and validation splits
# - input arguements : data_path - path to the train_data.csv file
# - output           : returns train and validation splits
def preprocess_split_train(data_path):
    if os.path.exists(data_path):
        loan_df = pd.read_csv(data_path)

        loan_df1 = pd.DataFrame.copy(loan_df)

        # the features chosen below are qualitatively examined and inferred that they are not of significant importance
        # for training
        loan_df1.drop(["member_id", 'funded_amnt_inv', 'grade', 'emp_title', 'desc', 'title'], axis=1, inplace=True)

        # features which contain more than 80% of null values are dropped
        loan_df1.dropna(thresh=loan_df1.shape[0] * 0.2, how='all', axis=1, inplace=True)

        # the features described below are inferred to categorical variables
        colname1 = ['term', 'sub_grade', 'pymnt_plan', 'purpose', 'zip_code',
                    'delinq_2yrs', 'inq_last_6mths', 'mths_since_last_delinq', 'open_acc', 'pub_rec',
                    'total_acc', 'initial_list_status', 'collections_12_mths_ex_med', 'mths_since_last_major_derog',
                    'application_type', 'addr_state', 'last_week_pay', 'acc_now_delinq', 'batch_enrolled', 'emp_length',
                    'home_ownership', 'verification_status']

        for k in loan_df1.keys():
            if k in colname1:
                # Categorical features which contain null values are replaced with most frequent entries
                # in the respective columns
                loan_df1[k].fillna(loan_df1[k].mode()[0], inplace=True)
            else:
                # Continuous features which contain null values are replaced with mean of the respective columns
                loan_df1[k].fillna(loan_df1[k].mean(), inplace=True)

        # X is the feature set and y is the label information
        X = loan_df1.drop('loan_status', axis=1)
        y = loan_df1['loan_status']

        # the overall data is split into train and validation splits
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                            random_state=42, stratify=y)

        le = {}
        for x in colname1:
            le[x] = LabelEncoder()
        # Categorical variables are one-hot encoded
        for x in colname1:
            X_test[x] = le[x].fit_transform(X_test.__getattr__(x))

        for x in colname1:
            X_train[x] = le[x].fit_transform(X_train.__getattr__(x))

        # All the features are scaled for uniform variance
        scaler = StandardScaler()
        scaler.fit_transform(X_train, y_train)
        scaler.fit_transform(X_test, y_test)

    else :
        raise Exception('Data path does not exist')

    return X_train, X_test, y_train, y_test


# preprocess_split_test: includes basic preprocessing operations required for the test data
# - input arguements : data_path - path to the train_data.csv file
# - output           : returns preprocessed test data
def preprocess_test(data_path):
    if os.path.exists(data_path):
        loan_df = pd.read_csv(data_path)

        loan_df1 = pd.DataFrame.copy(loan_df)

        loan_df1.drop(["member_id", 'funded_amnt_inv', 'grade', 'emp_title', 'desc', 'title'], axis=1, inplace=True)

        loan_df1.dropna(thresh=loan_df1.shape[0] * 0.2, how='all', axis=1, inplace=True)

        colname1 = ['term', 'sub_grade', 'pymnt_plan', 'purpose', 'zip_code',
                    'delinq_2yrs', 'inq_last_6mths', 'mths_since_last_delinq', 'open_acc', 'pub_rec',
                    'total_acc', 'initial_list_status', 'collections_12_mths_ex_med', 'mths_since_last_major_derog',
                    'application_type', 'addr_state', 'last_week_pay', 'acc_now_delinq', 'batch_enrolled', 'emp_length',
                    'home_ownership',
                    'verification_status']

        for k in loan_df1.keys():
            if k in colname1:
                loan_df1[k].fillna(loan_df1[k].mode()[0], inplace=True)
            else:
                loan_df1[k].fillna(loan_df1[k].mean(), inplace=True)

        X_test = loan_df1
        le = {}  # create blank dictionary
        for x in colname1:
            le[x] = LabelEncoder()

        for x in colname1:
            X_test[x] = le[x].fit_transform(X_test.__getattr__(x))

        scaler = StandardScaler()
        scaler.fit_transform(X_test)

    else:
        raise Exception('Data path does not exist')

    return X_test