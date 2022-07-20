from sklearn.linear_model import BayesianRidge

class CustomBayesianRidge:
    
    def __init__(self):
        pass
    
    def getRegressor(self, x_train, x_test, y_train, y_test):
        regressor = BayesianRidge().fit(x_train, y_train)
        return regressor
    
    def getName(self):
        return "Bayesian Ridge"
    
    def getDocumentationLink(self):
        return "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html#sklearn.linear_model.BayesianRidge"