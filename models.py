##########################################################
# The models.py file is used for configuring a classifier
# based on the user's choice
##########################################################

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression


# classifier_select: configures a classifier based on the choice
# - input arguements : choice of the classifier network
# - output           : returns the configured classifier
def classifier_select(choice):
    if choice == 'Log_Reg':
        classifier = LogisticRegression()
    elif choice == 'ADB_LR':
        classifier = AdaBoostClassifier(base_estimator=LogisticRegression())
    elif choice == 'DT':
        classifier = DecisionTreeClassifier()
    elif choice == 'RF':
        classifier = RandomForestClassifier(501)
    elif choice == 'KNN':
        classifier = KNeighborsClassifier(n_neighbors=10)
    elif choice == 'Ensemble':
        estimators = []
        model1 = LogisticRegression()
        estimators.append(('log', model1))
        model2 = RandomForestClassifier(501)
        estimators.append(('rf', model2))
        model3 = KNeighborsClassifier(n_neighbors=10)
        estimators.append(('KNN', model3))
        classifier = VotingClassifier(estimators)
    else :
        raise Exception('Invalid Classifier, please select a valid option from the list')
    return classifier