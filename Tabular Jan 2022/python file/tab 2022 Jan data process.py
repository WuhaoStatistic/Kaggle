import pandas as pd
import datetime
import math
import numpy as np

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

rowid = test['row_id']
test = test.drop(['row_id'], axis=1)

target = train['num_sold']
train = train.drop(['num_sold'], axis=1)


def day_of_year(year, mon, day):
    daylist1 = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    daylist2 = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if year % 4 == 0:
        res = sum([daylist1[x] for x in range(0, mon - 1)]) + day
    else:
        res = sum([daylist2[x] for x in range(0, mon - 1)]) + day
    return res


def is_weekend(mon, year, day):
    d1 = datetime.datetime(year, mon, day)
    d2 = datetime.datetime(2015, 1, 1)  # 1/1/2015 is Thursday we use this to calculate.
    daydif = (d1 - d2).days
    date = (daydif + 4) % 7
    if date == 0 or date == 6:
        return 1
    else:
        return 0


# we know 1/1/2015 is thursday.
def date_process(data):
    newyear = []  # 1 chrisms and new year 2 black friday 3 midsummer 4 normal
    weekend = []  # 1 weekend 2 weekdays
    dayofyear = []
    importance_factor = []
    annual_growth = []
    for i in range(data.shape[0]):
        strd = data.iloc[i, 0].split('/')
        mon = int(strd[0])
        day = int(strd[1])
        year = int(strd[2])
        dt = day_of_year(year, mon, day)
        if year == 2015:
            importance_factor.append(1)
        elif year == 2016:
            importance_factor.append(2)
        elif year == 2017:
            importance_factor.append(3)
        else:
            importance_factor.append(4)
        dayofyear.append(dt)
        # weekdend
        if is_weekend(mon, year, day) == 0:
            weekend.append(0)
        else:
            weekend.append(1)
        # newyear
        if (mon == 12 and day in [29, 30, 31]) or (mon == 1 and day in [1, 2]):  # 1 from 24/12 - 1/1
            newyear.append(1)
        else:
            newyear.append(0)
        # periodicity
        # annual growth
        if year == 2015:
            annual_growth.append(1)
        elif year == 2016:
            annual_growth.append(1.02)
        elif year == 2017:
            annual_growth.append(1.03)
        elif year == 2018:
            annual_growth.append(1.04)
        else:
            annual_growth.append(1.05)
    new = combine_new(data, [pd.Series(newyear), pd.Series(weekend), pd.Series(dayofyear), pd.Series(annual_growth),
                             pd.Series(importance_factor)])
    new.columns = ['date', 'country', 'store', 'product', 'newyear', 'weekend', 'dayofyear', 'annual_growth', 'impfac']
    new_df = pd.DataFrame()
    new_dummy = one_hot(new[['newyear', 'weekend', 'product', 'store', 'country']])
    new = new[['annual_growth', 'dayofyear', 'impfac']]
    new = pd.concat([new, new_dummy], axis=1)

    for k in range(1, 20):
        new_df[f'sin{k}'] = [np.sin(x / 365 * 2 * math.pi * k) for x in dayofyear]
        new_df[f'cos{k}'] = [np.cos(x / 365 * 2 * math.pi * k) for x in dayofyear]
        new_df[f'mug_sin{k}'] = new_df[f'sin{k}'] * new['product_Kaggle Mug']
        new_df[f'mug_cos{k}'] = new_df[f'cos{k}'] * new['product_Kaggle Mug']
        new_df[f'hat_sin{k}'] = new_df[f'sin{k}'] * new['product_Kaggle Hat']
        new_df[f'hat_cos{k}'] = new_df[f'cos{k}'] * new['product_Kaggle Hat']
    new_df = pd.concat([new, new_df], axis=1)
    res = pd.concat([data, new_df], axis=1)
    return res


def combine_new(df, new):
    frame = [i for i in new]
    con = pd.concat(frame, axis=1)
    frame = [df, con]
    df = pd.concat(frame, axis=1)
    return df


# df2 is the sub-dataframe that need to one-hot encoding
def one_hot(df2):
    res = pd.DataFrame()
    for i in df2.columns:
        temp = pd.get_dummies(df2[i], prefix=i)
        if len(res) == 0:
            res = temp
        else:
            res = res.join(temp)
    return res


new_train = date_process(train)
new_train = new_train.join(target, how='right')

new_test = date_process(test)
new_test = new_test.join(rowid, how='right')

new_train.to_csv('modified_train.csv')
new_test.to_csv('modified_test.csv')
