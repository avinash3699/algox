from sklearn.linear_model import MultiTaskElasticNet, MultiTaskElasticNetCV, MultiTaskLasso, MultiTaskLassoCV

class CustomMultiTaskElasticNet:
    
    def __init__(self):
        pass
    
    def getRegressor(self, x_train, x_test, y_train, y_test):
        regressor = MultiTaskElasticNet().fit(x_train, y_train)
        return regressor
    
    def getName(self):
        return "MultiTask Elastic Net"

    def getDocumentationLink(self):
        return "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.MultiTaskElasticNet.html"
    
class CustomMultiTaskElasticNetCV:
    
    def __init__(self):
        pass
    
    def getRegressor(self, x_train, x_test, y_train, y_test):
        regressor = MultiTaskElasticNetCV().fit(x_train, y_train)
        return regressor
    
    def getName(self):
        return "MultiTask Elastic Net CV"

    def getDocumentationLink(self):
        return "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.MultiTaskElasticNetCV.html"
    
class CustomMultiTaskLasso:
    
    def __init__(self):
        pass
    
    def getRegressor(self, x_train, x_test, y_train, y_test):
        regressor = MultiTaskLasso().fit(x_train, y_train)
        return regressor
    
    def getName(self):
        return "MultiTask Lasso"
    
    def getDocumentationLink(self):
        return "http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.MultiTaskLasso.html"
    
class CustomMultiTaskLassoCV:
    
    def __init__(self):
        pass
    
    def getRegressor(self, x_train, x_test, y_train, y_test):
        regressor = MultiTaskLassoCV().fit(x_train, y_train)
        return regressor
    
    def getName(self):
        return "MultiTask Lasso CV" 
    
    def getDocumentationLink(self):
        return "http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.MultiTaskLassoCV.html"