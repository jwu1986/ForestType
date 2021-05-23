# -*- coding:utf-8 -*-   `                                   
# J.W.                                                      
# May 2021
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn import model_selection
import pickle
import numpy as np
from sklearn import metrics
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='select_feature6.csv', type = str, help = 'input file')
parser.add_argument('--quicktrain', default = True, type = bool, help='whether girdsearch')
parser.add_argument('--output1', default='feature_important.csv', type = str, help = 'output file')
parser.add_argument('--output2', default='fin_result.csv', type = str, help = 'output file')

opt = parser.parse_args()

def main():
    '''
    quick_train: whether girdsearch
    ''' 
    quick_train = True
    
    print('loading data...')
    data = pd.read_csv(opt.input, header = None,low_memory=False)
    data1 = data.iloc[1:, 1:]
    data1.to_csv('head_reduce.csv', index = False)
    new_data = pd.read_csv('head_reduce.csv', header = None,low_memory = False)

    print('loaded data...')
    y_classed = list(pd.Categorical(new_data[0]).categories)
    y_new = pd.Categorical(new_data[0]).codes
    x_new = new_data.iloc[:, 1:]
    X_train, X_test, Y_train, Y_test = train_test_split(x_new, y_new, test_size=0.5, random_state=1, shuffle=True)

    NUM_TRIALS = 1
    non_nested_scores = np.zeros(NUM_TRIALS)
    nested_scores = np.zeros(NUM_TRIALS)

    rfc = RandomForestClassifier()
    p_grid = {"n_estimators": [100, 200, 300, 400, 500, 600], "max_depth": [5, 8, 15, 25, 30]}
    if opt.quicktrain:
        rfc = RandomForestClassifier(max_depth=15, n_estimators=200)
        rfc.fit(X_train, Y_train)
        with open("RFC_model.pkl", "wb") as f:
            pickle.dump(rfc, f)
        important_feature = rfc.feature_importances_
    else:
        bast_model = None
        bast_model_score = 0
        for i in range(NUM_TRIALS):
            # nest cv
            inner_cv = KFold(n_splits=4, shuffle=True, random_state=1)
            outer_cv = KFold(n_splits=4, shuffle=True, random_state=1)

            # Non_nested parameter search and scoring
            print('GridSearchCV...')
            rfc = GridSearchCV(estimator=rfc, param_grid=p_grid, cv=inner_cv)
            rfc.fit(X_train, Y_train)
            non_nested_scores[i] = rfc.best_score_
            print('non_nested_scores:', non_nested_scores[i],
                  '\tbast param:', rfc.best_params_)
            test_score = rfc.score(X_test, Y_test)
            if test_score > bast_model_score:
                bast_model_score = test_score
                bast_model = rfc.best_estimator_
            print('test score:', test_score)
            # Nested CV with parameter optimization
            nested_score = cross_val_score(rfc, X = X_train, y = Y_train, cv=outer_cv)
            print('nested_score:', nested_score, )
            nested_scores[i] = nested_score.mean()
            important_feature = rfc.best_estimator_.feature_importances_

        with open("RFC_model.pkl", "wb") as f:
            pickle.dump(bast_model, f)

        score_difference = non_nested_scores - nested_scores

        print("Average difference of {:6f} with std. dev. of {:6f}."
              .format(score_difference.mean(), score_difference.std()))

    y_hat = rfc.predict(X_test)
    
    mcm = multilabel_confusion_matrix(Y_test, y_hat)

    tp = mcm[:, 1, 1]
    tn = mcm[:, 0, 0]
    fn = mcm[:, 1, 0]
    fp = mcm[:, 0, 1]

    # 敏感性
    sn = tp / (tp + fn)
    sn_sum = tp.sum() / (tp.sum() + fn.sum())
    print('Sensibility in single subtype：', {yc: s for yc, s in zip(y_classed, sn)})
    print('Sensibility in total：', sn_sum)

    # 特异性
    sp = tn / (tn + fp)
    sp_sum = tn.sum() / (tn.sum() + fp.sum())
    print('Specificity in single subtype：', {yc: s for yc, s in zip(y_classed, sp)})
    print('Specificity in total', sp_sum)

    
    si = tp / (tp + fp)
    si_sum = tp.sum() / (tp.sum() + fp.sum())
    print('Precision in single subtype：', {yc: s for yc, s in zip(y_classed, si)})
    print('Precision in total', si_sum)

    # 精度
    acc = accuracy_score(Y_test, y_hat)
    print('accuracy:',acc)
     
    fi = pd.DataFrame(important_feature)
    fi.to_csv(opt.output1, header = False, index = False)
   
    older_data = pd.read_csv(opt.input,header = None,low_memory=False)

    x_data = older_data.iloc[1:, 2:].values
    y_data = older_data.iloc[1:, 0:2].values
    y_label = older_data.iloc[1:,1].values
    print("Silhouette coefficient:", metrics.silhouette_score(x_data, y_label, metric='euclidean'))

    feature_head = older_data.iloc[0, 2:].values
    indices = np.argsort(important_feature) # sort ascend
    sum = 0
    importances_greater_index = []
    for inx in indices:
        sum = sum + 1
        if sum <= 100:
            importances_greater_index.append(inx)
    x_data = x_data[:, importances_greater_index]
    
    feature_head = feature_head[importances_greater_index]
    
    data = np.hstack((y_data,x_data))
 
    feature_head = feature_head.tolist()
    
    feature_head.insert(0,'label')
    feature_head.insert(0,'Hugo_Symbol')
    
    new_data = pd.DataFrame(data.tolist(),columns = feature_head)
  
    new_data.to_csv(opt.output2, index = False)

if __name__ == '__main__':
    main()
