from sklearn.linear_model import LinearRegression

class Linear:
    
    def __init__(self):
        pass
    
    def getRegressor(self, x_train, x_test, y_train, y_test):
        regressor = LinearRegression().fit(x_train, y_train)
        return regressor
    
    def getName(self):
        return "Linear"
    
    def getDocumentationLink(self):
        return "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html"