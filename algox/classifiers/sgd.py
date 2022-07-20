from sklearn.linear_model import SGDClassifier

class SGD:
    
    def __init__(self):
        pass
    
    def getClassifier(self, x_train, x_test, y_train, y_test):
        classifier = SGDClassifier().fit(x_train, y_train)
        return classifier
    
    def getName(self):
        return "SGD"
    
    def getDocumentationLink(self):
        return "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier"