from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

class Logistic:
    
    def __init__(self):
        pass
    
    def getClassifier(self, x_train, x_test, y_train, y_test):
        classifier = LogisticRegression().fit(x_train, y_train)
        return classifier
    
    def getName(self):
        return "Logistic"
    
    def getDocumentationLink(self):
        return "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression"
    

class LogisticCV:
    
    def __init__(self):
        pass
    
    def getClassifier(self, x_train, x_test, y_train, y_test):
        classifier = LogisticRegressionCV().fit(x_train, y_train)
        return classifier
    
    def getName(self):
        return "LogisticCV"
    
    def getDocumentationLink(self):
        return "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html"