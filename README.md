# Credit Risk Modelling

This project describes the experimentation conducted with data collected over years for deciding whether a potential customer would default from paying back the load. The work describes appraoches based on machine learning for predicting the status of a loan. Different machine learning approaches are used in this study and AUC-ROC is considered as the metric of evaluation. 

The approaches used for experimentation and metrics:

| Algorithm | AUC-ROC |
| :---: | :---: |
| Logistic Regression(LR) | 50.56 |
| Boosted Logistic Regression | 59.61 |
| Decision Tree | 73.45 |
| Random Forest(RF)| 73.45 |
| K-Nearest Neighbors(KNN) | 61.85 |
| Ensemble (LR,RF,KNN) | 62.7 |

To run the code clone the project

1. train.py is used for training - path to training data is required
2. test.py is used for inference - path to test data and trained model file required

Trained models for the above experiments are available at:
https://drive.google.com/file/d/1HLquUEZ0iQDWMneXlTP5gZ54MSmFhxX9/view?usp=sharing 