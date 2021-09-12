# -*- coding:utf-8 -*-
# J.W.
# May 2021
# from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
def RF_feature_reduction_by_importance(feature_head, x, y, cnt):
    count = 1
    while(x.shape[1]>cnt):
        forest = RandomForestClassifier(n_estimators = 1000, random_state = 0, n_jobs = -1)
        forest.fit(x, y)
        score = forest.score(x,y)
        print('---',count,'---score:',score)
        count = count + 1
        importances = forest.feature_importances_
        indices = np.argsort(importances) # sort ascend
        importances_greater_index=[]
        for inx in indices:
            if (inx/x.shape[1]) > 0.1:
                importances_greater_index.append(inx)
        x = x[:, importances_greater_index]
        print((np.array(x)).shape)
        feature_head = feature_head[importances_greater_index]
    return feature_head, x, y
