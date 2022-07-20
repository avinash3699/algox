from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV

class CustomRidgeClassifier:
    
    def __init__(self):
        pass
    
    def getClassifier(self, x_train, x_test, y_train, y_test):
        classifier = RidgeClassifier().fit(x_train, y_train)
        return classifier
    
    def getName(self):
        return "Ridge"
    
    def getDocumentationLink(self):
        return "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html#sklearn.linear_model.RidgeClassifier"
    

class CustomRidgeCVClassifier:
    
    def __init__(self):
        pass
    
    def getClassifier(self, x_train, x_test, y_train, y_test):
        classifier = RidgeClassifierCV().fit(x_train, y_train)
        return classifier
    
    def getName(self):
        return "RidgeCV"
    
    def getDocumentationLink(self):
        return "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifierCV.html"