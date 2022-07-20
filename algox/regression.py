import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import warnings
warnings.filterwarnings('ignore')

from .regressors import (
    Linear,
    CustomLasso,
    CustomRidge,
    CustomRidgeCV,
    CustomLassoCV,
    CustomElasticNet,
    CustomElasticNetCV,
    CustomLars,
    CustomLarsCV,
    CustomLassoLars,
    CustomLassoLarsCV,
    CustomLassoLarsIC,
    CustomOrthogonalMatchingPursuit,
    CustomOrthogonalMatchingPursuitCV,
    CustomSGDRegressor,
    CustomMultiTaskElasticNet,
    CustomMultiTaskElasticNetCV,
    CustomMultiTaskLasso,
    CustomMultiTaskLassoCV,
    AutomaticRelevanceDetermination,
    CustomBayesianRidge,
    CustomPoissonRegressor,
    CustomTweedieRegressor,
    CustomGammaRegressor,
    CustomHuberRegressor,
    CustomRANSACRegressor,
    CustomTheilSenRegressor,
    CustomPassiveAggressiveRegressor,
    CustomSVR,
    CustomLinearSVR,
    CustomNuSVR,
    CustomDecisionTreeRegressor,
    CustomExtraTreeRegressor,
    CustomGaussianProcessRegressor,
    CustomKernelRidge,
    CustomMLPRegressor,
    CustomMultiOutputRegressor,
    CustomRegressorChain,
    CustomKNeighborsRegressor,
    CustomRadiusNeighborsRegressor,
    CustomLGBMRegressor,
    CustomCatBoostRegressor,
    CustomXGBRegressor
)


class Regression:
    
    algorithms = {
        "linear": Linear(),
        "ridge": CustomRidge(),
        "ridgeCV": CustomRidgeCV(),
        "lasso": CustomLasso(),
        "lassoCV": CustomLassoCV(),
        "elasticNet": CustomElasticNet(),
        "elasticNetCV": CustomElasticNetCV(),
        "lars": CustomLars(),
        "larsCV": CustomLarsCV(), 
        "lassoLars": CustomLassoLars(),
        "lassoLarsCV": CustomLassoLarsCV(), 
        "lassoLarsIC": CustomLassoLarsIC(),
        "orthogonal": CustomOrthogonalMatchingPursuit(),
        "orthogonalCV": CustomOrthogonalMatchingPursuitCV(),
        "sgd": CustomSGDRegressor(),
        "mten": CustomMultiTaskElasticNet(),
        "mtenCV": CustomMultiTaskElasticNetCV(),
        "mtl": CustomMultiTaskLasso(),
        "mtlCV": CustomMultiTaskLassoCV(),
        "relevanceDet": AutomaticRelevanceDetermination(),
        "bayesRidge": CustomBayesianRidge(),
        "poisson": CustomPoissonRegressor(),
        "tweedie": CustomTweedieRegressor(),
        "gamma": CustomGammaRegressor(),
        "huber": CustomHuberRegressor(),
        "ransac": CustomRANSACRegressor(),
        "theilSen": CustomTheilSenRegressor(),
        "passiveAggressive": CustomPassiveAggressiveRegressor(),
        "svr": CustomSVR(),
        "linearSvr": CustomLinearSVR(),
        "NuSvr": CustomNuSVR(),
        "decisionTree": CustomDecisionTreeRegressor(),
        "extraTree": CustomExtraTreeRegressor(),
        "gaussianProcess": CustomGaussianProcessRegressor(),
        "kernelRidge": CustomKernelRidge(),
        "mlp": CustomMLPRegressor(),
        "multiOutput": CustomMultiOutputRegressor(),
        "chain": CustomRegressorChain(),
        "knn": CustomKNeighborsRegressor(),
        "rn": CustomRadiusNeighborsRegressor(),
        "lgbm": CustomLGBMRegressor(),
        "catboost": CustomCatBoostRegressor(),
        "xgb": CustomXGBRegressor()
    }

    metricsTable = pd.DataFrame(columns = ["Algorithm", "MAE", "MSE", "RMSE", "RMSLE", "R2 Score"])
    
    def __init__(self, dataset:pd.DataFrame, test_size:float = 1/3):
        self.dataset = dataset 
        self.test_size = test_size
        self.splitDataset()
    
    def splitDataset(self):
        features = self.dataset.iloc[:, :-1]
        labels = self.dataset.iloc[:, -1]
        self.featuresTraining, self.featuresTesting, self.labelsTraining, self.labelsTesting = train_test_split(features, labels, test_size = self.test_size, random_state = 9)

    def getMetrics(self, algorithms:list = []):
        if not algorithms:
            algorithms = self.algorithms.values()
        else:
            algorithms = [self.algorithms[algo] for algo in algorithms]

        for algo in algorithms:
            try:
                regressor = algo.getRegressor(self.featuresTraining, self.featuresTesting, self.labelsTraining, self.labelsTesting)
                y_pred = regressor.predict(self.featuresTesting)
                self.updateMetricsTable(self.labelsTesting, y_pred, algo)
            except Exception as e:
                print(f"\nError - {algo.getName()}: {e}")
            
    def updateMetricsTable(self, y_true:pd.Series, y_pred:np.ndarray, algorithm:any):
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mse)
        rmsle = np.log(np.sqrt(mse))
        metrics = {"Algorithm" : algorithm.getName(), "MAE" : mae, "MSE" : mse, "RMSE" : rmse, "RMSLE" : rmsle, "R2 Score":r2}
            
        self.metricsTable = self.metricsTable.append(metrics, ignore_index=True)
    
    def displayMetricsTable(self, algorithms:list = [], sortBy:str = None, ascending:bool = False, download:bool = False):
        pd.options.display.max_columns = 6
        pd.options.display.max_colwidth = 35

        self.getMetrics(algorithms)

        if self.metricsTable.empty:
            return

        print()
        if(sortBy is not None):
            print(self.metricsTable.sort_values(sortBy, ascending=ascending))
        else:
            print(self.metricsTable)
        
        if(download):
             self.metricsTable.to_csv("metrics.csv", index = False)
             print("Download successful")

    def displayDocumentationLinks(self, algorithms:list = []):
        if not algorithms:
            # algorithms = self.algorithms
            algorithms = self.algorithms.values()
        else:
            algorithms = [self.algorithms[algo] for algo in algorithms]

        print()
        for algo in algorithms:
            print(f"{algo.getName()}: {algo.getDocumentationLink()}")