from sklearn.neural_network import MLPClassifier

class CustomMLPClassifier:
    
    def __init__(self):
        pass
    
    def getClassifier(self, x_train, x_test, y_train, y_test):
        classifier = MLPClassifier().fit(x_train, y_train)
        return classifier
    
    def getName(self):
        return "MLP Classifier"
    
    def getDocumentationLink(self):
        return "https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html"