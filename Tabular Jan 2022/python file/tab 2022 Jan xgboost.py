import xgboost as xgb
import pandas as pd
import os
from kaggle import tools
import time
from sklearn.preprocessing import scale

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_DEVICES'] = '0'
train = pd.read_csv('modified_train.csv')
test = pd.read_csv('modified_test.csv')
rowid = test['row_id']
test = test.drop(['row_id'], axis=1)
print(train.head(3))

rst = time.time()
train = tools.reduce_mem_usage(train)
target = train['num_sold']
train = train.drop(['num_sold'], axis=1)
train['dayofyear'] = scale(train['dayofyear'])

# def objective(trial, data=train, target=target):
#     X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=768, shuffle=False)
#     params = {
#         'max_depth': trial.suggest_int('max_depth', 4, 7),
#         'n_estimators': trial.suggest_int('n_estimators', 40, 4000),
#         'eta': trial.suggest_float('eta', 0.05, 0.1),
#         'subsample': trial.suggest_discrete_uniform('subsample', 0.6, 0.9, 0.1),
#         'colsample_bytree': trial.suggest_discrete_uniform('colsample_bytree', 0.6, 0.9, 0.1),
#         'min_child_weight': trial.suggest_loguniform('min_child_weight', 1e-5, 20),
#         'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-5, 100),
#         'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-5, 100),
#         'gamma': trial.suggest_loguniform('gamma', 1e-5, 100),
#         'predictor': "gpu_predictor",
#         'eval_metric': 'mape'
#     }
#
#     model = xgb.XGBRegressor(**params,
#                              tree_method='gpu_hist',
#                              booster='gbtree',
#                              random_state=768)
#     model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=50, verbose=False)
#     preds = model.predict(X_test)
#     score = tools.SMAPE(y_test, preds)
#
#     return score
#
#
# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=500)
# print('Number of finished trials:', len(study.trials))
# print('Best trial parameter:', study.best_trial.params)
# param = study.best_trial.params

param = {
    'max_depth': 4,
    'n_estimators': 404,
    'eta': 0.09999699847159464,
    'subsample': 0.8,
    'colsample_bytree': 0.6,
    'min_child_weight': 0.02028246995749663,
    'reg_lambda': 0.0002472970091804091,
    'reg_alpha': 0.0669574843764982,
    'gamma': 2.0525607113054525
}

model = xgb.XGBRegressor(**param)

# model.fit(train.iloc[0:17532, :], target[0:17532], verbose=False)
# preds = model.predict(train[17533:])
# real = target[17533:]
model.fit(train, target, verbose=False)
preds = model.predict(test)

dff = pd.DataFrame()
dff['row_id'] = rowid
dff['num_sold'] = preds
dff.to_csv('submit111.csv')
