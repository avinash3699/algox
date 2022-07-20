from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor

class CustomKNeighborsRegressor:
    
    def __init__(self):
        pass
    
    def getRegressor(self, x_train, x_test, y_train, y_test):
        regressor = KNeighborsRegressor().fit(x_train, y_train)
        return regressor
    
    def getName(self):
        return "K Neighbors"
    
    def getDocumentationLink(self):
        return "https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html#sklearn.neighbors.KNeighborsRegressor"
    
class CustomRadiusNeighborsRegressor:
    
    def __init__(self):
        pass
    
    def getRegressor(self, x_train, x_test, y_train, y_test):
        regressor = RadiusNeighborsRegressor().fit(x_train, y_train)
        return regressor
    
    def getName(self):
        return "Radius Neighbors"
    
    def getDocumentationLink(self):
        return "https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.RadiusNeighborsRegressor.html#sklearn.neighbors.RadiusNeighborsRegressor"