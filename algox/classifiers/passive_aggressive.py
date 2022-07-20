from sklearn.linear_model import PassiveAggressiveClassifier

class CustomPassiveAggressiveClassifier:
    
    def __init__(self):
        pass
    
    def getClassifier(self, x_train, x_test, y_train, y_test):
        classifier = PassiveAggressiveClassifier().fit(x_train, y_train)
        return classifier
    
    def getName(self):
        return "Passive Aggressive"
    
    def getDocumentationLink(self):
        return "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveClassifier.html#sklearn.linear_model.PassiveAggressiveClassifier"