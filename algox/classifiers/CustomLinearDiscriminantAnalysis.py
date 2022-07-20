from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class CustomLinearDiscriminantAnalysis:
    
    def __init__(self):
        pass
    
    def getClassifier(self, x_train, x_test, y_train, y_test):
        classifier = LinearDiscriminantAnalysis().fit(x_train, y_train)
        return classifier
    
    def getName(self):
        return "Linear Discriminant Analysis"
    
    def getDocumentationLink(self):
        return "https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html#sklearn.discriminant_analysis.LinearDiscriminantAnalysis"