##########################################################
# The train.py file is used for training credit risk model
##########################################################

import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score
import pickle
from models import classifier_select
from preprocess import preprocess_split_train
np.random.seed(42)
import os


# train_data describes the path of the input file in the form of .csv
train_data = r'C:\Users\Sriram Natarajan\Downloads\ML_Artivatic_dataset\train_indessa.csv'
# path variable describes the folder to save the trained model
model_save_path = r'F:\models'

# preprocess the input data - the details of preprocessing are available in preprocess.py
# - input arguements : train_data variable
# - output           : the entries in the .csv file are preprocessed and split into train and validation split
X_train, X_test, y_train, y_test = preprocess_split_train(train_data)

# Choice of the prediction algorithm: the choice variable could set from one of the options below
# Set 'Log_Reg' for Logistic Regression
# Set 'ADB_LR' for Boosted model with base model as Logistic Regression
# Set 'DT' for Decision Tree
# Set 'RF' for Random forest
# Set 'KNN' for K-Nearest Neighbours
# Set 'Ensemble' for ensemble of 3 networks: Log_reg, RF, KNN
choice = 'Log_Reg'

# classifier select function configures the model and returns it
# input arguements : choice of the algorithm
# output           : returns the configured classifier
classifier = classifier_select(choice)
classifier.fit(X_train,y_train)
Y_pred = classifier.predict(X_test)

# Metrics - Accuracy and AUC-ROC score
print(choice)
accuracy = round(classifier.score(X_test, y_test) * 100, 2)
print("Accuracy = ", accuracy)
print("ROC = ", roc_auc_score(y_test, Y_pred))

# Save the model
filename = os.path.join(model_save_path, choice + '.sav')
joblib.dump(classifier, open(filename, 'wb'))
