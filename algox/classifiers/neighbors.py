from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier, NearestCentroid

class CustomKNeighborsClassifier:
    
    def __init__(self):
        pass
    
    def getClassifier(self, x_train, x_test, y_train, y_test):
        classifier = KNeighborsClassifier().fit(x_train, y_train)
        return classifier
    
    def getName(self):
        return "K Neighbors Classifier"
    
    def getDocumentationLink(self):
        return "https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier"
    
    
class CustomRadiusNeighborsClassifier:
    
    def __init__(self):
        pass
    
    def getClassifier(self, x_train, x_test, y_train, y_test):
        classifier = RadiusNeighborsClassifier(radius=8.0).fit(x_train, y_train)
        return classifier
    
    def getName(self):
        return "Radius Neighbors Classifier"
    
    def getDocumentationLink(self):
        return "https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.RadiusNeighborsClassifier.html#sklearn.neighbors.RadiusNeighborsClassifier"
    
    
class CustomNearestCentroid:
    
    def __init__(self):
        pass
    
    def getClassifier(self, x_train, x_test, y_train, y_test):
        classifier = NearestCentroid().fit(x_train, y_train)
        return classifier
    
    def getName(self):
        return "Nearest Centroid"
    
    def getDocumentationLink(self):
        return "https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestCentroid.html#sklearn.neighbors.NearestCentroid"