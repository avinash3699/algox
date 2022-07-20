from sklearn.kernel_ridge import KernelRidge

class CustomKernelRidge:
    
    def __init__(self):
        pass
    
    def getRegressor(self, x_train, x_test, y_train, y_test):
        regressor = KernelRidge().fit(x_train, y_train)
        return regressor
    
    def getName(self):
        return "Kernel Ridge"
    
    def getDocumentationLink(self):
        return "https://scikit-learn.org/stable/modules/generated/sklearn.kernel_ridge.KernelRidge.html#sklearn.kernel_ridge.KernelRidge"