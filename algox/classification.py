import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    roc_auc_score
)
import warnings
warnings.filterwarnings('ignore')

from .classifiers import (
    Logistic,
    LogisticCV,
    CustomRidgeClassifier,
    CustomRidgeCVClassifier,
    SGD,
    PerceptronClassifier,
    CustomPassiveAggressiveClassifier,
    CustomLinearDiscriminantAnalysis,
    CustomQuadraticDiscriminantAnalysis,
    CustomSVC,
    CustomNuSVC,
    CustomLinearSVC,
    CustomCalibratedClassifierCV,
    CustomKNeighborsClassifier,
    CustomRadiusNeighborsClassifier,
    CustomNearestCentroid,
    CustomGaussianProcessClassifier,
    CustomGaussianNB,
    CustomMultinomialNB,
    CustomComplementNB,
    CustomBernoulliNB,
    CustomCategoricalNB,
    CustomDecisionTreeClassifier,
    CustomExtraTreeClassifier,
    CustomMLPClassifier
)


class Classification:

    algorithms = {
        "logistic": Logistic(),
        "ridge": CustomRidgeClassifier(),
        "ridgeCV": CustomRidgeCVClassifier(),
        "logisticCV": LogisticCV(),
        "sgd": SGD(),
        "perceptron": PerceptronClassifier(),
        "passiveAggressive": CustomPassiveAggressiveClassifier(),
        "linearDis": CustomLinearDiscriminantAnalysis(),
        "quadraticDis": CustomQuadraticDiscriminantAnalysis(),
        "svc": CustomSVC(),
        "nuSvc": CustomNuSVC(),
        "linearSvc": CustomLinearSVC(),
        "calibrated": CustomCalibratedClassifierCV(),
        "knn": CustomKNeighborsClassifier(),
        "rn": CustomRadiusNeighborsClassifier(),
        "centroid": CustomNearestCentroid(),
        "gaussianProcess": CustomGaussianProcessClassifier(),
        "gNB": CustomGaussianNB(),
        "mNB": CustomMultinomialNB(),
        "comNB": CustomComplementNB(),
        "bNB": CustomBernoulliNB(),
        "catNB": CustomCategoricalNB(),
        "decisionTree": CustomDecisionTreeClassifier(),
        "extraTree": CustomExtraTreeClassifier(),
        "mlp": CustomMLPClassifier()
    }
    
    metricsTable = pd.DataFrame(columns = ["Algorithm", "Accuracy", "Precision", "Recall", "F1 Score", "Log Loss", "AUC Score"])
    
    def __init__(self, dataset:pd.DataFrame, test_size:float = 1/3):
        self.dataset = dataset 
        self.test_size = test_size
        self.splitDataset()
    
    def splitDataset(self):
        features = self.dataset.iloc[:, :-1]
        labels = self.dataset.iloc[:, -1]
        self.featuresTraining, self.featuresTesting, self.labelsTraining, self.labelsTesting = train_test_split(features, labels, test_size = self.test_size, random_state = 369)
    
    def getMetrics(self, algorithms:list = []):
        if not algorithms:
            algorithms = self.algorithms.values()
        else:
            algorithms = [self.algorithms[algo] for algo in algorithms]

        for algo in algorithms:
            try:
                classifier = algo.getClassifier(self.featuresTraining, self.featuresTesting, self.labelsTraining, self.labelsTesting)
                y_pred = classifier.predict(self.featuresTesting)
                self.updateMetricsTable(self.labelsTesting, y_pred, algo)
            except Exception as e:
                print(f"\nError - {algo.getName()}: {e}")
    
    def updateMetricsTable(self, y_true:pd.Series, y_pred:np.ndarray, algorithm:any):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        logLoss = log_loss(y_true, y_pred)
        auc_score = roc_auc_score(y_true, y_pred)
        metrics = {"Algorithm" : algorithm.getName(), "Accuracy" : accuracy, "Precision" : precision, "Recall" : recall, "F1 Score" : f1, "Log Loss": logLoss, "AUC Score" : auc_score}
            
        self.metricsTable = self.metricsTable.append(metrics, ignore_index=True)
        
    def displayMetricsTable(self, algorithms:list = [], sortBy:str = None, ascending:bool = False, download:bool = False):
        pd.options.display.max_columns = 7
        pd.options.display.max_colwidth = 24

        self.getMetrics(algorithms)

        print()
        if(sortBy is not None):
            print(self.metricsTable.sort_values(sortBy, ascending = ascending))
        else:
            print(self.metricsTable)

        if(download):
             self.metricsTable.to_csv("metrics.csv", index = False)
             print("Download successful")
        
    def displayDocumentationLinks(self, algorithms = []):
        print()
        if not algorithms:
            algorithms = self.algorithms.values()
        else:
            algorithms = [self.algorithms[algo] for algo in algorithms]
        for algo in algorithms:
            print(algo.getName(), ":", algo.getDocumentationLink())