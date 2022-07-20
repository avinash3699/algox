from sklearn.svm import LinearSVC

class CustomLinearSVC:
    
    def __init__(self):
        pass
    
    def getClassifier(self, x_train, x_test, y_train, y_test):
        classifier = LinearSVC().fit(x_train, y_train)
        return classifier
    
    def getName(self):
        return "LinearSVC"
    
    def getDocumentationLink(self):
        return "https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC"