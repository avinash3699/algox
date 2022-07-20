from sklearn.linear_model import Lasso, LassoCV

class CustomLasso:
    
    def __init__(self):
        pass
    
    def getRegressor(self, x_train, x_test, y_train, y_test):
        regressor = Lasso().fit(x_train, y_train)
        return regressor
    
    def getName(self):
        return "Lasso"
    
    def getDocumentationLink(self):
        return "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html"
    
class CustomLassoCV:
    
    def __init__(self):
        pass
    
    def getRegressor(self, x_train, x_test, y_train, y_test):
        regressor = LassoCV().fit(x_train, y_train)
        return regressor
    
    def getName(self):
        return "Lasso CV"
    
    def getDocumentationLink(self):
        return "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html"