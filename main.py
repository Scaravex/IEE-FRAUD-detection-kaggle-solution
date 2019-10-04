# https://www.kaggle.com/vaishvik25/ieee-exploratory-data-analysis
# https://www.kaggle.com/amirhmi/a-comperhensive-guide-to-fraud-detection

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gc
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from feature_engineering import feature_engineering
from utils import reduce_mem_usage

gc.collect()

train_transaction = pd.read_csv('./input/train_transaction.csv', index_col='TransactionID')
test_transaction  = pd.read_csv('./input/test_transaction.csv', index_col='TransactionID')
train_identity    = pd.read_csv('./input/train_identity.csv', index_col='TransactionID')
test_identity  = pd.read_csv('./input/test_identity.csv', index_col='TransactionID')
sample_submission = pd.read_csv('./input/sample_submission.csv', index_col='TransactionID')
train_transaction = reduce_mem_usage(train_transaction)
test_transaction = reduce_mem_usage(test_transaction)
train_identity = reduce_mem_usage(train_identity)
test_identity = reduce_mem_usage(test_identity)
train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)
test  = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)
#test = pd.read_csv('final_test.csv',sep='\t')
#let's join both the transaction and identity
len(train_transaction)
len(train_identity)
len(test_transaction)
len(test_identity)
del train_transaction, train_identity
del test_transaction, test_identity


gc.collect()
# feature engineering and dropping unuseful columns
(train, test) = feature_engineering(train, test)
#test.to_csv('final_test.csv',sep='\t')
#train.to_csv('final_test.csv',sep='\t')
gc.collect()
train2 = train.copy()
test2 = test.copy()
#FILL NA 
#median for numerical variables
numerical_columns = list(test.select_dtypes(exclude=['object']).columns)

#it seems that adding -999 (a constant) improves the model --> try both
train[numerical_columns] = train[numerical_columns].fillna(-999)#train[numerical_columns].median())
test[numerical_columns] = test[numerical_columns].fillna(-999)#train[numerical_columns].median())
print("filling numerical columns null values done: ", len(numerical_columns))

#mode for categorical variables
categorical_columns = list(filter(lambda x: x not in numerical_columns, list(test.columns)))
categorical_columns[:5]
train[categorical_columns] = train[categorical_columns].fillna("-999")#train[categorical_columns].mode())
test[categorical_columns] = test[categorical_columns].fillna("-999")#train[categorical_columns].mode())
print("filling categorical columns null values done: ", len(categorical_columns))


#funzione per bucketizzare le categoriche. Così com'è tiene solo i livelli che contano per almeno 1%
#Questa invece la funzione per bucketizzare le categoriche. Così com'è tiene solo i livelli che contano per almeno 1%
def process_categorical_features(ftr_name, df_wrk):
    df = df_wrk.copy()
    df[ftr_name] = df[ftr_name].astype(str)
    keep_list = df[ftr_name].value_counts()/len(df)
    print(ftr_name+': starting with '+format(len(keep_list))+' different values')
    keep_list = keep_list[keep_list>.002].index.values
    df[ftr_name] = df[ftr_name].apply(lambda x: "other" if x not in keep_list else x)
    print(ftr_name+': ending with '+format(len(keep_list))+' different values')
    print()
    # Create levels
    return df
all_sub = pd.concat(train,test, axis=1)

for column in categorical_columns:
    print (column)
    train2 = process_categorical_features (column, train2)
    

#Encode categorical columns
from sklearn.preprocessing import LabelEncoder
for col in categorical_columns:
    le = LabelEncoder()
    le.fit(list(train[col].astype(str).values) + list(test[col].astype(str).values))
    train[col] = le.transform(list(train[col].astype(str).values))
    test[col] = le.transform(list(test[col].astype(str).values))

'''
#To save time: --> Eliminare con la submission FINALE!!!!!!  # Negative downsampling
gc.collect()
train_pos = train[train['isFraud']==1]
train_neg = train[train['isFraud']==0]
train_neg = train_neg.sample(int(train.shape[0] * 0.2), random_state=42)
train = pd.concat([train_pos,train_neg]).sort_index()
gc.collect()

'''

#sort_values descending
train = train.drop(['TransactionDT','card1'],axis = 1)
test  = test.drop(['TransactionDT','card1'], axis = 1)

labels = train["isFraud"]
train.drop(["isFraud"], axis=1, inplace=True)
X_train, y_train = train, labels
gc.collect()
del train
del test
lgb_submission=sample_submission.copy()
lgb_submission['isFraud'] = 0


#5 fold cross-validation
n_fold = 5
folds = KFold(n_fold)
ROC_Avg = 0

params = {'num_leaves': 491,
          'min_child_weight': 0.03454472573214212,
          'feature_fraction': 0.3797454081646243,
          'bagging_fraction': 0.4181193142567742,
          'min_data_in_leaf': 106,
          'objective': 'binary',
          'max_depth': -1,
          'learning_rate': 0.006883242363721497,
          "boosting_type":"gbdt",#'goss'
          "bagging_seed": 11,
          "metric": 'auc',
          "verbosity": -1,
          'reg_alpha': 0.3899927210061127,
          'reg_lambda': 0.6485237330340494,
          'random_state': 47
          #'num_threads':10
          #'device' :'gpu',
          #'is_unbalance':True
          #'scale_pos_weight':9
         }
a = 3
#random split not correct --> should be ordered by timestamp! (Clauser)
for fold_n, (train_index, valid_index) in enumerate(folds.split(X_train)):
    print(fold_n)
    
    X_train_, X_valid = X_train.iloc[train_index], X_train.iloc[valid_index]
    y_train_, y_valid = y_train.iloc[train_index], y_train.iloc[valid_index]
    dtrain = lgb.Dataset(X_train, label=y_train,categorical_feature = categorical_columns)
    dvalid = lgb.Dataset(X_valid, label=y_valid,categorical_feature = categorical_columns)
    
    lgbclf = lgb.LGBMClassifier(
            num_leaves= 512,
            n_estimators=512,
            max_depth=9,
            learning_rate=0.0069,
            feature_fraction= 0.3797454081646243,
            bagging_fraction = 0.4181193142567742,
            subsample=0.85,
            metric = "auc",
            colsample_bytree=0.85,
            boosting_type= "gbdt",
            reg_alpha=0.4,
            reg_lamdba=0.6
    )    
    
    X_train_, X_valid = X_train.iloc[train_index], X_train.iloc[valid_index]
    y_train_, y_valid = y_train.iloc[train_index], y_train.iloc[valid_index]
    lgbclf.fit(X_train_,y_train_)
    
    del X_train_,y_train_
    print('finish train')
    pred=lgbclf.predict_proba(test)[:,1]
    val=lgbclf.predict_proba(X_valid)[:,1]
    print('finish pred')
    del lgbclf, X_valid
    ROC_auc = roc_auc_score(y_valid, val)
    print('ROC accuracy: {}'.format(ROC_auc))
    ROC_Avg = ROC_Avg + ROC_auc
    del val,y_valid
    lgb_submission['isFraud'] = lgb_submission['isFraud']+ pred/n_fold
    del pred
    gc.collect()

print(ROC_Avg/n_fold)
lgb_submission.insert(0, "TransactionID", np.arange(3663549, 3663549 + 506691))
lgb_submission.to_csv('prediction.csv', index=False)
#ROC 90.43%

'''
import matplotlib.pyplot as plt
# Plot feature importance
feature_importance = lgbclf.feature_importances_
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
sorted_idx = sorted_idx[len(feature_importance) - 50:]
pos = np.arange(sorted_idx.shape[0]) + .5

plt.figure(figsize=(10,12))
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, X_train.columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()
'''

###Alternative LGB
feature_importance_df = pd.DataFrame()
i = 0
folds =5
kf = KFold(n_splits = folds, shuffle = True, random_state=50)
for tr_idx, val_idx in kf.split(X_train, y_train):

    X_tr = X_train.iloc[tr_idx, :]
    y_tr = y_train.iloc[tr_idx]
    d_train = lgb.Dataset(X_tr, label=y_tr,categorical_feature = categorical_columns)

    model = lgb.train(params,
                      train_set=d_train,
                      num_boost_round=200,
                      verbose_eval=200)
        
    
    yvalid = model.predict(test) / folds
    
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = X_tr.columns
    fold_importance_df["importance"] = model.feature_importance()
    fold_importance_df["fold"] = i + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)  
    i+=1
    del X_tr,d_train