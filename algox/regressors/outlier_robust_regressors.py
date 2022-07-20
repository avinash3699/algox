from sklearn.linear_model import HuberRegressor, RANSACRegressor, TheilSenRegressor

class CustomHuberRegressor:
    
    def __init__(self):
        pass
    
    def getRegressor(self, x_train, x_test, y_train, y_test):
        regressor = HuberRegressor().fit(x_train, y_train)
        return regressor
    
    def getName(self):
        return "Huber"
    
    def getDocumentationLink(self):
        return "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor.html#sklearn.linear_model.HuberRegressor"
    
class CustomRANSACRegressor:
    
    def __init__(self):
        pass
    
    def getRegressor(self, x_train, x_test, y_train, y_test):
        regressor = RANSACRegressor().fit(x_train, y_train)
        return regressor
    
    def getName(self):
        return "RANSAC"
    
    def getDocumentationLink(self):
        return "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RANSACRegressor.html#sklearn.linear_model.RANSACRegressor"
    
class CustomTheilSenRegressor:
    
    def __init__(self):
        pass
    
    def getRegressor(self, x_train, x_test, y_train, y_test):
        regressor = TheilSenRegressor().fit(x_train, y_train)
        return regressor
    
    def getName(self):
        return "Theil Sen"
    
    def getDocumentationLink(self):
        return "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TheilSenRegressor.html#sklearn.linear_model.TheilSenRegressor"