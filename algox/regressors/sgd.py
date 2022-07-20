from sklearn.linear_model import SGDRegressor

class CustomSGDRegressor:
    
    def __init__(self):
        pass
    
    def getRegressor(self, x_train, x_test, y_train, y_test):
        regressor = SGDRegressor().fit(x_train, y_train)
        return regressor
    
    def getName(self):
        return "SGD"
    
    def getDocumentationLink(self):
        return "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html#sklearn.linear_model.SGDRegressor"