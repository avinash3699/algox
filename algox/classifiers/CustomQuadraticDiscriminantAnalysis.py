from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
class CustomQuadraticDiscriminantAnalysis:
    
    def __init__(self):
        pass
    
    def getClassifier(self, x_train, x_test, y_train, y_test):
        classifier = QuadraticDiscriminantAnalysis().fit(x_train, y_train)
        return classifier
    
    def getName(self):
        return "Quadratic Discriminant Analysis"
    
    def getDocumentationLink(self):
        return "https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html#sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis"