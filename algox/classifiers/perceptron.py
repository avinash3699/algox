from sklearn.linear_model import Perceptron

class PerceptronClassifier:
    
    def __init__(self):
        pass
    
    def getClassifier(self, x_train, x_test, y_train, y_test):
        classifier = Perceptron().fit(x_train, y_train)
        return classifier
    
    def getName(self):
        return "Perceptron"
    
    def getDocumentationLink(self):
        return "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html#sklearn.linear_model.Perceptron"