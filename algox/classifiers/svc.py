from sklearn.svm import SVC

class CustomSVC:
    
    def __init__(self):
        pass
    
    def getClassifier(self, x_train, x_test, y_train, y_test):
        classifier = SVC().fit(x_train, y_train)
        return classifier
    
    def getName(self):
        return "SVC"
    
    def getDocumentationLink(self):
        return "https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC"