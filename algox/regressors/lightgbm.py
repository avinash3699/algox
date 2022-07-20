from lightgbm import LGBMRegressor

class CustomLGBMRegressor:
    
    def __init__(self):
        pass
    
    def getRegressor(self, x_train, x_test, y_train, y_test):
        regressor = LGBMRegressor().fit(x_train, y_train)
        return regressor
    
    def getName(self):
        return "Light GBM"
    
    def getDocumentationLink(self):
        return "https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html"