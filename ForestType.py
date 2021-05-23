# -*- coding:utf-8 -*-
# J.W
# May 2021 
import pandas as pd
from RF_feature_reduction_by_importance import RF_feature_reduction_by_importance
from kmeans_select import kmeans_select
from sklearn.cluster import KMeans
from sklearn import metrics
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='data_expression_median_T2.csv', type = str, help = 'input file')
parser.add_argument('--cluster', default = 6, type = int, help = 'k-means')
parser.add_argument('--feature', default = 500, type = int, help = 'the number of importance feature')
parser.add_argument('--output', default = 'select_feature6.csv', type = str, help = 'output file')
opt = parser.parse_args()

if __name__=='__main__':
   
    print('loading data...')
    data_df = pd.read_csv(opt.input, header=None,low_memory=False)
    print('loaded...')
    data_df_dropna = data_df.dropna(axis=1, how='any')  # drop all cols that have any NaN values
    # data_df_dropna = data_df_dropna.dropna(axis=0, how='any')  # drop all rows that have any NaN values
    indx_all = data_df_dropna.iloc[1:, 0].values
    x, y = data_df_dropna.iloc[1:, 2:].values, data_df_dropna.iloc[1:, 1].values # 0 col is number 1 col is label 2: cols are data
    feature_head = data_df_dropna.iloc[0, 2:].values
    # y_new = pd.Categorical(y).codes
    remain_feature_cnt = opt.feature
    feature_head_reduced, x_reduced, y_reduced = RF_feature_reduction_by_importance(feature_head, x,y,remain_feature_cnt) # remain the first 99% features every time
    print('features reduce...')

    num_of_cluster = opt.cluster
    kmeans_more = KMeans(n_clusters=num_of_cluster)  # n_clusters:number of cluster
    kmeans_more.fit(x_reduced)
   
    selected_label = kmeans_select(kmeans_more.labels_, y_reduced)
    print('labels select...')
    new_x = []  # reduce x, if label exist, x remains
    new_y = []
    new_indx = []
    for i in range(len(selected_label)):
        if selected_label[i] != 'delete':
            new_x.append(x_reduced[i,:])    # it is 2-D
            new_y.append([selected_label[i]])  # to generate 2-Dimension
            new_indx.append([indx_all[i]])
    #print('Feature Reduced!')
    # print(np.array(new_y).shape)
    # print(np.array(new_x).shape)
    data = np.hstack((np.array(new_y),np.array(new_x)))
    # print((np.array(new_y)).shape)
    data = np.hstack((np.array(new_indx),data))
    feature_head_reduced = feature_head_reduced.tolist()
    feature_head_reduced.insert(0,'label')
    feature_head_reduced.insert(0,'Hugo_Symbol')
    new_data = pd.DataFrame(data.tolist(),columns=feature_head_reduced)
    new_data.to_csv(opt.output, index = False)
