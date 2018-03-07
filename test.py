# -*- coding:utf-8 -*-
#test analysis statistic values and calculate running time
import pandas as pd
from wrap_up import *
from wrap_up_cal_time import eda_analysis_cal_time

##0.Read Data##
df = pd.read_csv('train.csv')
label = df['TARGET']
df = df.drop(['ID', 'TARGET'], axis=1)

##1.EDA##
df_eda_summary = eda_analysis(missSet=[np.nan, 9999999999, -999999], df=df.iloc[:, 0:3])
print(df_eda_summary)
df_eda_summary = eda_analysis_cal_time(missSet=[np.nan, 9999999999, -999999], df=df)

##2. Calculating Running time##
import timeit

start = timeit.default_timer()
df_eda_summary = eda_analysis(missSet=[np.nan, 9999999999, -999999], df=df.iloc[:, 0:3])
print "EDA Running time:{0:.2f} seconds".format(timeit.default_timer()-start)