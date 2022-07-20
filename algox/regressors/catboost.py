from catboost import CatBoostRegressor

class CustomCatBoostRegressor:
    
    def __init__(self):
        pass
    
    def getRegressor(self, x_train, x_test, y_train, y_test):
        regressor = CatBoostRegressor(logging_level = 'Silent').fit(x_train, y_train)
        return regressor
    
    def getName(self):
        return "Cat Boost"
    
    def getDocumentationLink(self):
        return "https://catboost.ai/en/docs/concepts/python-reference_catboostregressor"