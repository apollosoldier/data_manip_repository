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

## BEG Models ##

import lightgbm as lgbm
from catboost import CatBoostRegressor
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
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



def check_na_ratio(df):
  empty_features = []
  missing_ratio = {}
  for _ in features:
    if df[_].isna().sum() >0:
      print("{0} has {1} NaN value(s) ON {2} = {3}% missing".format(_,df[_].isna().sum(), 
                                                                df[_].value_counts().sum()+df[_].isna().sum(), 
                                                                100*np.round(
                                                                    df[_].isna().sum()/
                                                                    (df[_].value_counts().sum()+df[_].isna().sum()
                                                                    ), 3)))
      missing_ratio[_] = 100*np.round(df[_].isna().sum()/ (df[_].value_counts().sum()+df[_].isna().sum()
                                                                    ), 3)
    if 100*np.round(df[_].isna().sum()/ (df[_].value_counts().sum()+df[_].isna().sum()
                                                                    ), 3)==100:
      empty_features.append(_)
    else:
      print("**Feature named** {0} has none missing value".format(_))
