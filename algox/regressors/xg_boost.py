from xgboost import XGBRegressor

class CustomXGBRegressor:
    
    def __init__(self):
        pass
    
    def getRegressor(self, x_train, x_test, y_train, y_test):
        regressor = XGBRegressor(logging_level = 'Silent').fit(x_train, y_train)
        return regressor
    
    def getName(self):
        return "XG Boost"
    
    def getDocumentationLink(self):
        return "https://xgboost.readthedocs.io/en/stable/python/python_api.html#module-xgboost.sklearn"