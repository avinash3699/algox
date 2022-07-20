from sklearn.gaussian_process import GaussianProcessRegressor

class CustomGaussianProcessRegressor:
    
    def __init__(self):
        pass
    
    def getRegressor(self, x_train, x_test, y_train, y_test):
        regressor = GaussianProcessRegressor().fit(x_train, y_train)
        return regressor
    
    def getName(self):
        return "Gaussian Process"
    
    def getDocumentationLink(self):
        return "https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html#sklearn.gaussian_process.GaussianProcessRegressor"