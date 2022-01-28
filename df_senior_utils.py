import os
import gc
import sys

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
import lightgbm as lgb
from catboost import CatBoostRegressor
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

## END Models ##
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline


my_models = [GradientBoostingClassifier(),
             lgbm(), CatBoostClassifier(), XGBClassifier(),
             RandomForestClassifier(), GaussianMixture()]


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
