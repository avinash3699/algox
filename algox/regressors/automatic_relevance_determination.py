from sklearn.linear_model import ARDRegression

class AutomaticRelevanceDetermination:
    
    def __init__(self):
        pass
    
    def getRegressor(self, x_train, x_test, y_train, y_test):
        regressor = ARDRegression().fit(x_train, y_train)
        return regressor
    
    def getName(self):
        return "Automatic Relevance Determination"
    
    def getDocumentationLink(self):
        return "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ARDRegression.html#sklearn.linear_model.ARDRegression"