from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB, CategoricalNB

class CustomGaussianNB:
    
    def __init__(self):
        pass
    
    def getClassifier(self, x_train, x_test, y_train, y_test):
        classifier = GaussianNB().fit(x_train, y_train)
        return classifier
    
    def getName(self):
        return "GaussianNB"
    
    def getDocumentationLink(self):
        return "https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn.naive_bayes.GaussianNB"
    
    
    
class CustomMultinomialNB:
    
    def __init__(self):
        pass
    
    def getClassifier(self, x_train, x_test, y_train, y_test):
        classifier = MultinomialNB().fit(x_train, y_train)
        return classifier
    
    def getName(self):
        return "MultinomialNB"
    
    def getDocumentationLink(self):
        return "https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html#sklearn.naive_bayes.MultinomialNB"
    
    
class CustomComplementNB:
    
    def __init__(self):
        pass
    
    def getClassifier(self, x_train, x_test, y_train, y_test):
        classifier = ComplementNB().fit(x_train, y_train)
        return classifier
    
    def getName(self):
        return "ComplementNB"
    
    def getDocumentationLink(self):
        return "https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.ComplementNB.html#sklearn.naive_bayes.ComplementNB"
    
    
class CustomBernoulliNB:
    
    def __init__(self):
        pass
    
    def getClassifier(self, x_train, x_test, y_train, y_test):
        classifier = BernoulliNB().fit(x_train, y_train)
        return classifier
    
    def getName(self):
        return "BernoulliNB"
    
    def getDocumentationLink(self):
        return "https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html#sklearn.naive_bayes.BernoulliNB"
    
    
class CustomCategoricalNB:
    
    def __init__(self):
        pass
    
    def getClassifier(self, x_train, x_test, y_train, y_test):
        classifier = CategoricalNB().fit(x_train, y_train)
        return classifier
    
    def getName(self):
        return "CategoricalNB"
    
    def getDocumentationLink(self):
        return "https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.CategoricalNB.html#sklearn.naive_bayes.CategoricalNB"