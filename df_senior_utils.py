import os
import gc
import sys
from time import time

import random
import numpy as np
import pandas as pd
import seaborn as sns

from tqdm.notebook import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
tqdm_notebook().pandas()

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 500)
warnings.simplefilter("ignore")
from scipy.stats import chi2_contingency
## BEG Models ##

#import lightgbm as lgbm
#from catboost import CatBoostRegressor
#from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
#from xgboost import XGBClassifier
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
## END Models ##
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.model_selection import GridSearchCV
#from lightgbm import LGBMClassifier as lgbm
from sklearn.neural_network import MLPClassifier
#from catboost import CatBoost
#from mlxtend.plotting import plot_decision_regions
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


#==============================================================================================
def lgbm_features_importance(clf,features,n=15,size=(15,12)):
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df=fold_importance_df.sort_values("importance",ascending=False).iloc[:n,:]
    plt.figure(figsize=size)
    sns.barplot(x="importance", y="Feature", data=fold_importance_df) 
    plt.title('Features importance ')
    plt.tight_layout()
    plt.show()
#==============================================================================================
def fit_lgbm(config,dtrain,dval):
    
    trn_data=lgb.Dataset(dtrain[config.features], label=dtrain[config.target],categorical_feature=config.categoricals)
    val_data=lgb.Dataset(dval[config.features], label=dval[config.target],categorical_feature=config.categoricals)
    clf=lgb.train(config.param, trn_data, 5_000_000, valid_sets = [trn_data, val_data],
                 verbose_eval=config.vb_eval, early_stopping_rounds = config.es)
    
    pred_train = clf.predict(dtrain[config.features],num_iteration=clf.best_iteration) 
    pred_oof = clf.predict(dval[config.features],num_iteration=clf.best_iteration)

    lgbm_features_importance(clf,config.features,n=30,size=(15,12))

    return clf,pred_train,pred_oof

def fit(model, X, y,i):
  print("Model : ",str(model)+str(i))
  start = time()
  cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=8, random_state=0)
  score = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
  speed[str(model)] = np.round(time()-start, 2)
  accuracy[str(model)] = np.mean(score).round(3)
  print(
      f"Mean accuracy: {accuracy[str(model)]}\nStd: {np.std(score):.3f}\nRun time: {speed[str(model)]}s"
  )
  print("\n\n\n\t")
  try:
    print("Predict proba", str(model.predict_proba(X_test))+str(i))
  except:
    print("Can't predict likelihood for this model")
  return accuracy

def catboost_features_importance(clf,features,n=15,size=(15,12)):
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df=fold_importance_df.sort_values("importance",ascending=False).iloc[:n,:]
    plt.figure(figsize=size)
    sns.barplot(x="importance", y="Feature", data=fold_importance_df) 
    plt.title('Features importance ')
    plt.tight_layout()
    plt.show()

def check_na_ratio(df):
    empty_features = []
    missing_ratio = {}
    features = df.columns.tolist()
    for feature in features:
        if df[feature].isna().sum() > 0:
            ratio = 100 * np.round(
                df[feature].isna().sum()
                / (df[feature].value_counts().sum() + df[feature].isna().sum()),
                3,
            )
            print(
                f"{feature} has {df[feature].isna().sum()} NaN value(s) ON {df[feature].value_counts().sum()+df[feature].isna().sum()} = {ratio}% missing"
            )
            missing_ratio[feature] = ratio
            if ratio == 100:
                empty_features.append(feature)
        else:
            print(f"**Feature named** {feature} has none missing value")
    return missing_ratio, empty_features


