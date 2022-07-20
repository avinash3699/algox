from sklearn.calibration import CalibratedClassifierCV

class CustomCalibratedClassifierCV:
    
    def __init__(self):
        pass
    
    def getClassifier(self, x_train, x_test, y_train, y_test):
        classifier = CalibratedClassifierCV().fit(x_train, y_train)
        return classifier
    
    def getName(self):
        return "Calibrated Classifier CV"
    
    def getDocumentationLink(self):
        return "https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html#sklearn.calibration.CalibratedClassifierCV"