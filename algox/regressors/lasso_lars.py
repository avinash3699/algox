from sklearn.linear_model import LassoLars, LassoLarsCV, LassoLarsIC

class CustomLassoLars:
    
    def __init__(self):
        pass
    
    def getRegressor(self, x_train, x_test, y_train, y_test):
        regressor = LassoLars().fit(x_train, y_train)
        return regressor
    
    def getName(self):
        return "Lasso Lars"
    
    def getDocumentationLink(self):
        return "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLars.html#sklearn.linear_model.LassoLars"
    
class CustomLassoLarsCV:
    
    def __init__(self):
        pass
    
    def getRegressor(self, x_train, x_test, y_train, y_test):
        regressor = LassoLarsCV().fit(x_train, y_train)
        return regressor
    
    def getName(self):
        return "Lasso Lars CV"
    
    def getDocumentationLink(self):
        return "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLarsCV.html#sklearn.linear_model.LassoLarsCV"
    
class CustomLassoLarsIC:
    
    def __init__(self):
        pass
    
    def getRegressor(self, x_train, x_test, y_train, y_test):
        regressor = LassoLarsIC().fit(x_train, y_train)
        return regressor
    
    def getName(self):
        return "Lasso Lars IC"
    
    def getDocumentationLink(self):
        return "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLarsIC.html#sklearn.linear_model.LassoLarsIC"