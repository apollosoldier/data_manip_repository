%pylab inline
import pandas as pd


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
