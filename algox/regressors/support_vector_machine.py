from sklearn.svm import LinearSVR, SVR, NuSVR

class CustomLinearSVR:
    
    def __init__(self):
        pass
    
    def getRegressor(self, x_train, x_test, y_train, y_test):
        regressor = LinearSVR().fit(x_train, y_train)
        return regressor
    
    def getName(self):
        return "Linear SVR"
    
    def getDocumentationLink(self):
        return "https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html#sklearn.svm.LinearSVR"
    
class CustomSVR:
    
    def __init__(self):
        pass
    
    def getRegressor(self, x_train, x_test, y_train, y_test):
        regressor = SVR().fit(x_train, y_train)
        return regressor
    
    def getName(self):
        return "SVR"
    
    def getDocumentationLink(self):
        return "https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html?highlight=svr#sklearn.svm.SVR"
    
class CustomNuSVR:
    
    def __init__(self):
        pass
    
    def getRegressor(self, x_train, x_test, y_train, y_test):
        regressor = NuSVR().fit(x_train, y_train)
        return regressor
    
    def getName(self):
        return "Nu SVR"
    
    def getDocumentationLink(self):
        return "https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVR.html#sklearn.svm.NuSVR"