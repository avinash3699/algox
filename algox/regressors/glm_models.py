from sklearn.linear_model import PoissonRegressor, TweedieRegressor, GammaRegressor

class CustomPoissonRegressor:
    
    def __init__(self):
        pass
    
    def getRegressor(self, x_train, x_test, y_train, y_test):
        regressor = PoissonRegressor().fit(x_train, y_train)
        return regressor
    
    def getName(self):
        return "Poisson"
    
    def getDocumentationLink(self):
        return "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PoissonRegressor.html#sklearn.linear_model.PoissonRegressor"
    
class CustomTweedieRegressor:
    
    def __init__(self):
        pass
    
    def getRegressor(self, x_train, x_test, y_train, y_test):
        regressor = TweedieRegressor().fit(x_train, y_train)
        return regressor
    
    def getName(self):
        return "Tweedie"
    
    def getDocumentationLink(self):
        return "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TweedieRegressor.html#sklearn.linear_model.TweedieRegressor"
    
class CustomGammaRegressor:
    
    def __init__(self):
        pass
    
    def getRegressor(self, x_train, x_test, y_train, y_test):
        regressor = GammaRegressor().fit(x_train, y_train)
        return regressor
    
    def getName(self):
        return "Gamma"
    
    def getDocumentationLink(self):
        return "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.GammaRegressor.html#sklearn.linear_model.GammaRegressor"