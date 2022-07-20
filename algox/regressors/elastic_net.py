from sklearn.linear_model import ElasticNet, ElasticNetCV

class CustomElasticNet:
    
    def __init__(self):
        pass
    
    def getRegressor(self, x_train, x_test, y_train, y_test):
        regressor = ElasticNet().fit(x_train, y_train)
        return regressor
    
    def getName(self):
        return "Elastic Net"

    def getDocumentationLink(self):
        return "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html#sklearn-linear-model-elasticnet"
    
class CustomElasticNetCV:
    
    def __init__(self):
        pass
    
    def getRegressor(self, x_train, x_test, y_train, y_test):
        regressor = ElasticNetCV().fit(x_train, y_train)
        return regressor
    
    def getName(self):
        return "Elastic Net CV"
    
    def getDocumentationLink(self):
        return "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNetCV.html#sklearn.linear_model.ElasticNetCV"