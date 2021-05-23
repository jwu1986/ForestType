# -*- coding:utf-8 -*-
# J.W.
# May 2021
# from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
def RF_feature_reduction_by_importance(feature_head, x, y, cnt):
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    count = 1
    while(x.shape[1]>cnt):
    # for i in range(k):
        forest = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
        forest.fit(x, y)
        score = forest.score(x,y)
        print('---',count,'---score:',score)
        count = count + 1
        importances = forest.feature_importances_
        indices = np.argsort(importances) # sort ascend
        sum =0
        importances_greater_index=[]
        for inx in indices:
            sum = sum + 1
            if (sum/x.shape[1]) > 0.01:
                importances_greater_index.append(inx)
        x = x[:, importances_greater_index]
        feature_head = feature_head[importances_greater_index]
    return feature_head, x, y
