# -*- coding:utf-8 -*-
# different statistic values expression and calculation
from __future__ import division
import pandas as pd
import numpy as np
from scipy import stats


##0.Read Data##
##dataset:https://www.kaggle.com/c/santander-customer-satisfaction
df = pd.read_csv('train.csv')
label = df['TARGET']
df = df.drop(['ID', 'TARGET'], axis=1)

##1.Basic Analysis##
#(1)Missing Value#
missSet = [np.nan, 9999999999, -999999]

#(2)Count distinct#
##len(df.iloc[:, 0].unique())

count_un = df.iloc[:, 0:3].apply(lambda x:len(x.unique()))

#(3)Zero Values#
##np.sum(df.iloc[:,0] == 0)

count_zero = df.iloc[:, 0:3].apply(lambda x:np.sum(x == 0))

#(4)Mean Values#
np.mean(df.iloc[:, 0]) # 没有去除缺失值之前的均值很低

df.iloc[:,0][~np.isin(df.iloc[:,0], missSet)] # 去除缺失值
np.mean(df.iloc[:,0][~np.isin(df.iloc[:,0], missSet)]) # 去除缺失值后的均值计算

df_mean = df.iloc[:,0:3].apply(lambda x:np.mean(x[~np.isin(x, missSet)]))

#(5)Median Values#
np.median(df.iloc[:,0])

df.iloc[:,0][~np.isin(df.iloc[:,0], missSet)]# 去除缺失值
np.median(df.iloc[:,0][~np.isin(df.iloc[:,0], missSet)])# 去除缺失值后的均值计算

df_median = df.iloc[:,0:3].apply(lambda x:np.median(x[~np.isin(x, missSet)]))

#(6)Mode Values#第0-2列中，众数本身是谁
df_mode = df.iloc[:,0:3].apply(lambda x: stats.mode(x[~np.isin(x, missSet)])[0][0])

#(7)Mode Percentage#第0-2列中，众数出现的比例有多高
df_mode_count = df.iloc[:,0:3].apply(lambda x: stats.mode(x[~np.isin(x, missSet)])[1][0])

df_mode_perct = df_mode_count/df.shape[0]

#(8)Min Values#
np.min(df.iloc[:,0])

df.iloc[:,0][~np.isin(df.iloc[:,0], missSet)]# 去除缺失值
np.min(df.iloc[:,0][~np.isin(df.iloc[:,0], missSet)])# 去除缺失值之后进行最小值计算

df_min = df.iloc[:, 0:3].apply(lambda x:np.min(x[~np.isin(x, missSet)]))

#(9)Max Values#
np.max(df.iloc[:,0])

df.iloc[:,0][~np.isin(df.iloc[:,0], missSet)]# 去除缺失值
np.max(df.iloc[:,0][~np.isin(df.iloc[:,0], missSet)])# 去除缺失值之后进行最大值计算

df_max = df.iloc[:, 0:3].apply(lambda x:np.max(x[~np.isin(x, missSet)]))


#(10)quantile values 分位数
np.percentile(df.iloc[:,0], (1,5,25,50,75,95,99))

df.iloc[:,0][~np.isin(df.iloc[:,0], missSet)]# 去除缺失值
np.percentile(df.iloc[:,0][~np.isin(df.iloc[:,0], missSet)], (1,5,25,50,75,95,99))

json_quantile = {}

for i,name in enumerate(df.iloc[:,0:3].columns):
    print('the %d columns: %s' %(i,name))
    json_quantile[name] = np.percentile(df[name][~np.isin(df[name], missSet)], (1,5,25,50,75,95,99))

df_quantife = pd.DataFrame(json_quantile)[df.iloc[:,0:3].columns].T #指定名字放入

#(11)Frequent Values
df.iloc[:,0].value_counts().iloc[0:5,]

df.iloc[:,0][~np.isin(df.iloc[:,0], missSet)]# 去除缺失值
df.iloc[:,0][~np.isin(df.iloc[:,0], missSet)].value_counts()[0:5,]## 去除缺失值之后进行（前五位）频数的统计

json_fre_name = {}
json_fre_count = {}


def fill_fre_top_5(x):
    if len(x) <= 5:
        new_array = np.full(5, np.nan)
        new_array[0:len(x)] = x
        return new_array


df['ind_var1_0'].value_counts()
len(df['imp_sal_var16_ult1'].value_counts())

for i, name in enumerate(df[['ind_var1_0', 'imp_sal_var16_ult1']].columns):
    #1.index name
    index_name = df[name][~np.isin(df[name], missSet)].value_counts().iloc[0:5, ].index.values
    # if the length of arrary is less than 5
    index_name = fill_fre_top_5(index_name)
    #store result
    json_fre_name[name] = index_name

    #2. values count
    values_count = df[name][~np.isin(df[name], missSet)].value_counts().iloc[0:5, ].values
    # if the length of arrary is less than 5
    values_count = fill_fre_top_5(values_count)
    #store result
    json_fre_count[name] = values_count

df_fre_name = pd.DataFrame(json_fre_name)[df[['ind_var1_0', 'imp_sal_var16_ult1']].columns].T
df_fre_count = pd.DataFrame(json_fre_count)[df[['ind_var1_0', 'imp_sal_var16_ult1']].columns].T

df_fre = pd.concat([df_fre_name,df_fre_count],axis=1)

#(12)Miss Values
np.sum(np.isin(df.iloc[:,0], missSet)) #统计缺失值
df_miss = df.iloc[:,0:3].apply(lambda x:np.sum(np.isin(x, missSet))) #遍历每一个遍历的缺失值情况