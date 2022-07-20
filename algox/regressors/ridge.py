from sklearn.linear_model import Ridge, RidgeCV

class CustomRidge:
    
    def __init__(self):
        pass
    
    def getRegressor(self, x_train, x_test, y_train, y_test):
        regressor = Ridge().fit(x_train, y_train)
        return regressor
    
    def getName(self):
        return "Ridge"
    
    def getDocumentationLink(self):
        return "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html"
    
class CustomRidgeCV:
    
    def __init__(self):
        pass
    
    def getRegressor(self, x_train, x_test, y_train, y_test):
        regressor = RidgeCV().fit(x_train, y_train)
        return regressor
    
    def getName(self):
        return "Ridge CV"
    
    def getDocumentationLink(self):
        return "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html#sklearn.linear_model.RidgeCV"