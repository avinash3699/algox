from scipy.stats import randint
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

class CustomDecisionTreeClassifier:
    
    def __init__(self):
        pass
    
    def getClassifier(self, x_train, x_test, y_train, y_test):
        classifier = DecisionTreeClassifier().fit(x_train, y_train)
        return classifier
    
    def getName(self):
        return "Decision Tree Classifier"
    
    def getDocumentationLink(self) -> str:
        return "https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier"
    
    def getBestParams(self, classifier, x_train, x_test, y_train, y_test):
    
        param_dict = {
            #'max_depth': [2, 3, 5, 10, 20],
            #'min_samples_leaf': [5, 10, 20, 50, 100],
            #'criterion': ["gini", "entropy"]
            
            'max_depth': list(range(-10, 10)),
            'min_samples_leaf': list(range(-10, 10)),
            'criterion': ["gini", "entropy"]
        }
        cv_classifier = GridSearchCV(classifier, param_dict, cv = 5, scoring = 'f1_macro')
        cv_classifier.fit(x_train, y_train)
        return cv_classifier.best_estimator_, cv_classifier.best_score_

class CustomExtraTreeClassifier:
    
    def __init__(self):
        pass
    
    def getClassifier(self, x_train, x_test, y_train, y_test):
        classifier = ExtraTreeClassifier().fit(x_train, y_train)
        return classifier
    
    def getName(self):
        return "Extra Tree Classifier"
    
    def getDocumentationLink(self):
        return "https://scikit-learn.org/stable/modules/generated/sklearn.tree.ExtraTreeClassifier.html#sklearn.tree.ExtraTreeClassifier"
 