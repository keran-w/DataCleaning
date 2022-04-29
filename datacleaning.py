"""
常用数据清洗方法整合模块

这个模块将现实应用场景中的一些数据清洗方法做了整合，基于Numpy和Pandas开发，
在只需要Python基础语法的情况下，依然可以完成一些相对复杂的数据清洗任务。

作者: 王可冉
版本: 2022年4月29日
"""

import os
import numpy as np
import pandas as pd
from io import StringIO
from xlsx2csv import Xlsx2csv

from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder

import warnings
warnings.filterwarnings('ignore')


# 读取文件
def read_file(filename, columns=None, sheetid=0, sep=','):

    # filename：读取文件名
    # columns：选择被保留的列
    # 返回：读取的DataFrame

    filename = filename.replace('\\', '/')
    filetype = filename.split('.')[-1]
    if filetype == 'csv':
        data = pd.read_csv(filename, sep=sep)
    elif filetype == 'xlsx':
        buffer = StringIO()
        Xlsx2csv(filename, outputencoding='utf-8').convert(buffer, sheetid)
        buffer.seek(0)
        data = pd.read_csv(buffer)
    elif filetype == 'xls':
        data = pd.read_excel(buffer)
    elif filetype == 'txt':
        f = open(filename, 'r', encoding='utf-8-sig')
        data = f.read().split(sep)
        f.close()

    if columns is not None and filetype in ('csv', 'xlsx', 'xls'):
        data = data[columns]
    return data



def merge(df1, df2, left_on=None, right_on=None, drop_duplaicates=True):
    # 合并两个表
    
    # df1, df2: 将被合并的两个表，将df2合并到df1中
    # left_on， right_on：自定义左右两边用于合并的列，默认为列名相同的
    # 返回：合并后的DataFrame
    
    df_merge = df1.merge(df2, 'left') if left_on is None \
        else df1.merge(df2, 'left', left_on=left_on, right_on=right_on)

    df_merge.drop_duplicates(inplace=drop_duplaicates)
    return df_merge

def sift(data, col_name, tgt_list):
    return data.query(f'{col_name} in @tgt_list')

def drop_columns(data, cols):
    return data.drop(cols, 1)

# 分类数据变哑变量
def cat2ohe(X, cat):
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoder_df = pd.DataFrame(encoder.fit_transform(X[[cat]]).toarray(), columns=[f'{cat}_{i}' for i in sorted(X[cat].unique())]).astype('int')
    return encoder_df

# 含有分隔符的分类数据变哑变量
def cat2ohe_split(X, cat, delimiter='+'):
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoder_df = pd.DataFrame(encoder.fit_transform(X[[cat]]).toarray(), columns=[f'{cat}_{i}' for i in sorted(X[cat].unique())]).astype('int')
    return encoder_df

# 基础表
def base_table_process_helper(results, row, values, num_values, first_index, index_count):
    i, n, v = row[0], row[1], row[2:]
    flag = 1
    try: results[n][i][0]
    except: flag = 0
    
    if flag == 0:
        results[n][i] = v[0]
        for j in range(num_values):
            results[n + f'_{values[j]}'][i] = v[j + 1]
    else:
        k = index_count[n + i]
        results[n][first_index[i] + k] = v[0]
        for j in range(num_values):
            results[n + f'_{values[j]}'][first_index[i] + k] = v[j + 1]
        index_count[n + i] += 1

def base_table_process(data_csv, id, name, key, values):
    data_csv = data_csv.query(f'{name} == {name}')[[id, name, key] + values].drop_duplicates()
    data_csv[id] = data_csv[id].astype('string')
    num_values = len(values)
    new_indices = []
    for _, (i, count) in data_csv.groupby([id, name], sort=False).count().max(level=0).max(1).reset_index().iterrows():
        new_indices += [i] * count

    value_count = len(values) + 1
    new_columns = data_csv[name].unique().repeat(value_count)
    for i in range(1, value_count):
        new_columns[i::value_count] += f'_{values[i - 1]}'

    results = pd.DataFrame('', index=new_indices, columns=new_columns).astype(object)
    first_index = {id:row_num for _, (row_num, id) in pd.DataFrame(results.index).drop_duplicates().reset_index().iterrows()}
    index_count = {nameid:0 for nameid in (data_csv[name] + data_csv[id]).unique()}

    for _, row in tqdm(data_csv[[id, name, key] + values].iterrows(), total=data_csv.shape[0]):
        base_table_process_helper(results, row, values, num_values, first_index, index_count)

    return results

def build_base_table(data, id, name, key, values, output_filename):
    results = base_table_process(data, id, name, key, values)
    results.index.name = id
    results = results.reset_index().fillna('')
    results.to_csv(output_filename, index=False, encoding='utf-8-sig')
    return results

def rank_time(data_, time_col, other_cols, ascending=True):

    data = data_.copy()
    data[time_col] = pd.to_datetime(data[time_col])
    all_cols = other_cols + [time_col]
    data = data.sort_values(all_cols)
    data['rank'] = data[all_cols].groupby(other_cols, sort=False)[time_col].rank(ascending=ascending, method='first')
    return data


def remove_negative_cost(data, time_col, cost_col, other_cols):

    del_list = []
    prev_idx = 0
    for idx, row in data[data.duplicated(other_cols, keep=False)].sort_values(other_cols + [time_col]).iterrows():
        if row[cost_col] < 0:
            del_list += [prev_idx, idx]
        prev_idx = idx
    return data.drop(del_list)

def remove_empty_cells(data, col):
    """
        去除数据中在某一列中为空值的所有数据
        
    
    """
    return data.query(f'{col} == {col}')