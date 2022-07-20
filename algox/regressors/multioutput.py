from sklearn.multioutput import MultiOutputRegressor, RegressorChain
from sklearn.linear_model import Ridge, LogisticRegression

class CustomMultiOutputRegressor:
    
    def __init__(self):
        pass
    
    def getRegressor(self, x_train, x_test, y_train, y_test):
        regressor = MultiOutputRegressor(Ridge(random_state=123)).fit(x_train, y_train)
        return regressor
    
    def getName(self):
        return "Multi Output"

    def getDocumentationLink(self):
        return "http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputRegressor.html"
    
class CustomRegressorChain:
    
    def __init__(self):
        pass
    
    def getRegressor(self, x_train, x_test, y_train, y_test):
        logreg = LogisticRegression(solver='lbfgs',multi_class='multinomial')
        regressor = RegressorChain(base_estimator=logreg, order=[0, 1]).fit(x_train, y_train)
        return regressor
    
    def getName(self):
        return "Regressor Chain"
    
    def getDocumentationLink(self):
        return "http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.RegressorChain.html"