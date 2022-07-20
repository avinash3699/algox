from sklearn.neural_network import MLPRegressor

class CustomMLPRegressor:
    
    def __init__(self):
        pass
    
    def getRegressor(self, x_train, x_test, y_train, y_test):
        regressor = MLPRegressor().fit(x_train, y_train)
        return regressor
    
    def getName(self):
        return "Perceptron"
    
    def getDocumentationLink(self):
        return "https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html#sklearn.neural_network.MLPRegressor"