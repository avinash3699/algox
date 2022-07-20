from sklearn.linear_model import Lars, LarsCV

class CustomLars:
    
    def __init__(self):
        pass
    
    def getRegressor(self, x_train, x_test, y_train, y_test):
        regressor = Lars().fit(x_train, y_train)
        return regressor
    
    def getName(self):
        return "Lars"
    
    def getDocumentationLink(self):
        return "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lars.html#sklearn.linear_model.Lars"
    
class CustomLarsCV:
    
    def __init__(self):
        pass
    
    def getRegressor(self, x_train, x_test, y_train, y_test):
        regressor = LarsCV().fit(x_train, y_train)
        return regressor
    
    def getName(self):
        return "Lars CV"
    
    def getDocumentationLink(self):
        return "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LarsCV.html#sklearn.linear_model.LarsCV"