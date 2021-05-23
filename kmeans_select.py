# -*- coding:utf-8 -*-
# J.W
# May 2021
import pandas as pd
def kmeans_select(kmeans_labels,y):
    # 对原始label 和kmeans label 去重
    kmeans_labels2 = []
    y_labels2 = []
    for i in kmeans_labels:
        if i not in kmeans_labels2:
            kmeans_labels2.append(i)
    for i in y:
        if i not in y_labels2:
            y_labels2.append(i)
    #  二维列表统计原始label 在kmeans label中的对应数量
    df_labels = pd.DataFrame(columns=kmeans_labels2,index=y_labels2)
    df_labels.loc[:, :] = 0
    for i in range(len(y)):
        val = df_labels[kmeans_labels[i]][y[i]]
        df_labels[kmeans_labels[i]][y[i]] = val + 1
    # 找到 原始label 在kmeans label中最大的那一类
    km_argmax = {}
    for row in df_labels.index:
        s1 = df_labels.loc[row,:]
        # 获得 s1 中最大值的索引
        km_argmax[row]=(s1[s1 == s1.max()].index[0],s1.max())
#######################################################################################
    # 对原始label中重复选择了kmeans label的部分重新分配kmeans中的label
    new_km = km_argmax.copy()
    selected_klabel = []
    for key1,val1 in km_argmax.items():
        selected_klabel.extend([val1[0]])
        for key2,val2 in km_argmax.items():
            if key1 != key2:
                if (val1[0] == val2[0]) and (val1[1] > val2[1]):
                    new_km[key2] = (-1,val2[1])
    selected_klabel = list(set(selected_klabel))
    for key,val in new_km.items():
        if val[0] == -1:
            remain_label = kmeans_labels2.copy()
            for label_tmp in selected_klabel:
                remain_label.remove(label_tmp)
            ss = df_labels.loc[key, remain_label]
            klabel = ss[ss == ss.max()].index[0]
            new_km[key] = (klabel, ss.max())
            selected_klabel.append(klabel)
################################################################################
    # 把kmeans label中与原始label不一致的置空‘’
    new_label = []
    for i in range(len(kmeans_labels)):
        if kmeans_labels[i] in selected_klabel:
            if new_km[y[i]][0]==kmeans_labels[i]:
                new_label.append(kmeans_labels[i])  #  一致的保留
            else:
                new_label.append('delete')   #  不一致的置为, 'delete'
        else:
            new_label.append(kmeans_labels[i])    # 新label 保留
    return new_label
