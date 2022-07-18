import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifierCV
from warnings import simplefilter
from avaliacao import confusionMatrixDetails
simplefilter(action='ignore', category=FutureWarning)

def DecisionTree(X_train, y_train, X_test, y_test):
    classifier1 = DecisionTreeClassifier(criterion='entropy', max_depth=200, min_samples_split=10, min_samples_leaf=10)
    classifier1.fit(X_train, y_train)
    y_predDT = classifier1.predict(X_test)
    precisao, revocacao, f1, acuracia, especificidade = confusionMatrixDetails(y_test, y_predDT, "DT")
    return precisao, revocacao, f1, acuracia, especificidade, np.array(y_predDT),classifier1
    
def KNeighbors(X_train, y_train, X_test, y_test):    
    classifier3 = KNeighborsClassifier()
    classifier3.fit(X_train, y_train)
    y_predKNN = classifier3.predict(X_test)
    precisao, revocacao, f1, acuracia, especificidade = confusionMatrixDetails(y_test, y_predKNN, "kNN")
    return precisao, revocacao, f1, acuracia, especificidade, np.array(y_predKNN)
    
def RandomForest(X_train, y_train, X_test, y_test): 
    classifier5 = RandomForestClassifier(n_estimators=500, max_depth=200, min_samples_split=10)
    classifier5.fit(X_train, y_train)
    y_predRF = classifier5.predict(X_test)
    precisao, revocacao, f1, acuracia, especificidade = confusionMatrixDetails(y_test, y_predRF, "RF")
    return precisao, revocacao, f1, acuracia, especificidade, np.array(y_predRF),classifier5
    
def Ridge(X_train, y_train, X_test, y_test): 
    classifier8 = RidgeClassifierCV()
    classifier8.fit(X_train, y_train)
    y_predRidge = classifier8.predict(X_test)
    precisao, revocacao, f1, acuracia, especificidade = confusionMatrixDetails(y_test, y_predRidge, "Ridge")    
    return precisao, revocacao, f1, acuracia, especificidade, np.array(y_predRidge)