from sklearn.linear_model import OrthogonalMatchingPursuit, OrthogonalMatchingPursuitCV

class CustomOrthogonalMatchingPursuit:
    
    def __init__(self):
        pass
    
    def getRegressor(self, x_train, x_test, y_train, y_test):
        regressor = OrthogonalMatchingPursuit().fit(x_train, y_train)
        return regressor
    
    def getName(self):
        return "Orthogonal Matching Pursuit"
    
    def getDocumentationLink(self):
        return "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.OrthogonalMatchingPursuit.html#sklearn.linear_model.OrthogonalMatchingPursuit"
    
class CustomOrthogonalMatchingPursuitCV:
    
    def __init__(self):
        pass
    
    def getRegressor(self, x_train, x_test, y_train, y_test):
        regressor = OrthogonalMatchingPursuitCV().fit(x_train, y_train)
        return regressor
    
    def getName(self):
        return "Orthogonal Matching Pursuit CV"
    
    def getDocumentationLink(self):
        return "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.OrthogonalMatchingPursuitCV.html#sklearn.linear_model.OrthogonalMatchingPursuitCV"