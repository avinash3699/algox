from sklearn.gaussian_process import GaussianProcessClassifier

class CustomGaussianProcessClassifier:
    
    def __init__(self):
        pass
    
    def getClassifier(self, x_train, x_test, y_train, y_test):
        classifier = GaussianProcessClassifier().fit(x_train, y_train)
        return classifier
    
    def getName(self):
        return "Gaussian Process Classifier"
    
    def getDocumentationLink(self):
        return "https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html#sklearn.gaussian_process.GaussianProcessClassifier"