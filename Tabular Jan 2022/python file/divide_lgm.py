import lightgbm as lgb
import os
import pandas as pd
import time
from kaggle import tools
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_DEVICES'] = '0'

hat = pd.read_csv('hat.csv')
target_hat = hat['num_sold']
hat = hat.drop(['num_sold'], axis=1)
hat = hat.drop(['impfac'], axis=1)

mug = pd.read_csv('mug.csv')
target_mug = mug['num_sold']
mug = mug.drop(['num_sold'], axis=1)
mug = mug.drop(['impfac'], axis=1)

sti = pd.read_csv('sticker.csv')
target_sti = sti['num_sold']
sti = sti.drop(['num_sold'], axis=1)
sti = sti.drop(['impfac'], axis=1)
# train = pd.read_csv('modified_train.csv')
print(hat.columns)

rst = time.time()
# train = tools.reduce_mem_usage(train)
# weight = train['impfac']
# target = train['num_sold']
# train = train.drop(['num_sold'], axis=1)
# train = train.drop(['impfac'], axis=1)
test = pd.read_csv('modified_test.csv')
rowid = test['row_id']
test = test.drop(['row_id'], axis=1)
hat_test = pd.read_csv('hat_test.csv')
mug_test = pd.read_csv('mug_test.csv')
sti_test = pd.read_csv('sti_test.csv')
hat_test = hat_test.drop(['row_id'], axis=1)
mug_test = mug_test.drop(['row_id'], axis=1)
sti_test = sti_test.drop(['row_id'], axis=1)
# hat
params_hat = {
    'learning_rate': 0.1,

    'n_estimators': 90,
    'num_leaves': 20,
    'max_depth': 9,

    'min_child_samples': 6,
    'min_child_weight': 0.005,

    'colsample_bytree': 0.4,
    'subsample': 0.009,
    'reg_alpha': 1.07,
    'reg_lambda': 7.5,

}

# mug
params_mug = {
    'learning_rate': 0.1,

    'n_estimators': 90,
    'num_leaves': 14,
    'max_depth': 6,

    'min_child_samples': 6,
    'min_child_weight': 0.005,

    'colsample_bytree': 0.4,
    'subsample': 0.009,
    'reg_alpha': 1.07,
    'reg_lambda': 7.5,

}
params_sti = {
    'learning_rate': 0.1,

    'n_estimators': 65,
    'num_leaves': 25,
    'max_depth': 4,

    'min_child_samples': 2,
    'min_child_weight': 0.005,

    'colsample_bytree': 0.4,
    'subsample': 0.009,
    'reg_alpha': 11,
    'reg_lambda': 9,

}
# smape = make_scorer(tools.SMAPE, greater_is_better=False)
# # cv_param = {'n_estimators': [60, 65, 70, 75], 'num_leaves': [25, 20, 30],
# #             'max_depth': [4]}
# cv_param = {'min_child_samples': [2], 'min_child_weight': [0.005],
#             'colsample_bytree': [0.4], 'subsample': [0.1],
#             'reg_alpha': [11], 'reg_lambda': [9]}
#
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
model_lgb = lgb.LGBMRegressor(**params_hat)
model_lgb.fit(hat, target_hat)
res_lgb = model_lgb.predict(hat_test)

model_lgb2 = lgb.LGBMRegressor(**params_mug)
model_lgb2.fit(mug, target_mug)
res_lgb2 = model_lgb2.predict(mug_test)

model_lgb3 = lgb.LGBMRegressor(**params_sti)
model_lgb3.fit(sti, target_sti)
res_lgb3 = model_lgb3.predict(sti_test)

res = []
for i in range(len(res_lgb)):
    res.append(res_lgb2[i])
    res.append(res_lgb[i])
    res.append(res_lgb3[i])

dff = pd.DataFrame()
dff['row_id'] = rowid
dff['num_sold'] = pd.Series(res)
dff.to_csv('submit.csv')
