import lightgbm as lgb
import os
import pandas as pd
import time
from kaggle import tools
from sklearn.metrics import make_scorer

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_DEVICES'] = '0'
train = pd.read_csv('modified_train.csv')
rst = time.time()
train = tools.reduce_mem_usage(train)
weight = train['impfac']
target = train['num_sold']
train = train.drop(['num_sold'], axis=1)
train = train.drop(['impfac'], axis=1)
test = pd.read_csv('modified_test.csv')
rowid = test['row_id']
test = test.drop(['row_id'], axis=1)
params1 = {
    'learning_rate': 0.1,

    'n_estimators': 94,
    'num_leaves': 40,
    'max_depth': 6,

    'min_child_samples': 6,
    'min_child_weight': 0.005,

    'colsample_bytree': 0.5,
    'subsample': 0.009,
    'reg_alpha': 17,
    'reg_lambda': 1.6,

}
smape = make_scorer(tools.SMAPE, greater_is_better=False)
# cv_param = {'n_estimators': [40, 50, 80, 60, 70], 'num_leaves': [40, 50],
#             'max_depth': [6, 7]}
# cv_param = {'min_child_samples': [6, 10, 20], 'min_child_weight': [0.005, 0.01, 0.1, 0.5],
#             'colsample_bytree': [0.1, 0.5, 0.8], 'subsample': [0.1, 0.3],
#             'reg_alpha': [1, 10, 20], 'reg_lambda': [1, 5, 10]}
# cv_param = {'colsample_bytree': [500, 1000, 2000], 'subsample': 20}
# cv_param = {'reg_alpha': [500, 1000, 2000], 'reg_lambda': 20}
# model = lgb.LGBMRegressor(**params1)
# cv_lgb = GridSearchCV(estimator=model, param_grid=cv_param, cv=5,
#                       verbose=1, scoring=smape,
#                       n_jobs=-1)

# cv_lgb.fit(train, target)
# best_score = cv_lgb.best_score_
# best_pa = cv_lgb.best_params_
# ret = time.time()
# print('loading data costs ' + tools.timetrans(ret - rst))
# print(best_pa)
# print(best_score)
model_lgb = lgb.LGBMRegressor(**params1)
model_lgb.fit(train, target)
res_lgb = model_lgb.predict(test)
dff = pd.DataFrame()
dff['row_id'] = rowid
dff['num_sold'] = res_lgb
dff.to_csv('submit.csv')
