from sklearn.svm import NuSVC

class CustomNuSVC:
    
    def __init__(self):
        pass
    
    def getClassifier(self, x_train, x_test, y_train, y_test):
        classifier = NuSVC().fit(x_train, y_train)
        return classifier
    
    def getName(self):
        return "NuSVC"
    
    def getDocumentationLink(self):
        return "https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVC.html#sklearn.svm.NuSVC"