from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor

class CustomDecisionTreeRegressor:
    
    def __init__(self):
        pass
    
    def getRegressor(self, x_train, x_test, y_train, y_test):
        regressor = DecisionTreeRegressor().fit(x_train, y_train)
        return regressor
    
    def getName(self):
        return "Decision Tree"
    
    def getDocumentationLink(self):
        return "https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html?highlight=decisiontreeregressor#sklearn.tree.DecisionTreeRegressor"
    
class CustomExtraTreeRegressor:
    
    def __init__(self):
        pass
    
    def getRegressor(self, x_train, x_test, y_train, y_test):
        regressor = ExtraTreeRegressor().fit(x_train, y_train)
        return regressor
    
    def getName(self):
        return "Extra Tree"
    
    def getDocumentationLink(self):
        return "https://scikit-learn.org/stable/modules/generated/sklearn.tree.ExtraTreeRegressor.html#sklearn.tree.ExtraTreeRegressor"