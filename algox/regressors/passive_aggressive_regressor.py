from sklearn.linear_model import PassiveAggressiveRegressor

class CustomPassiveAggressiveRegressor:
    
    def __init__(self):
        pass
    
    def getRegressor(self, x_train, x_test, y_train, y_test):
        regressor = PassiveAggressiveRegressor().fit(x_train, y_train)
        return regressor
    
    def getName(self):
        return "Passive Aggressive"
    
    def getDocumentationLink(self):
        return "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveRegressor.html#sklearn.linear_model.PassiveAggressiveRegressor"