"""
常用数据清洗方法整合模块

这个模块将现实应用场景中的一些数据清洗方法做了整合，基于Numpy和Pandas开发，
只需要掌握Python的基础语法，就可以完成一些相对复杂的数据清洗任务。

作者: 王可冉
版本: 2022年5月6日
""" 

import os, sys, re
from typing import List
import numpy as np
import pandas as pd
from io import StringIO
from xlsx2csv import Xlsx2csv

from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder

import warnings
warnings.filterwarnings('ignore')


def read_file(filename: str, columns=None, sheetid=1, sep=','):
    """ 读取多种文件类型的数据，对于列表数据，可以选择部分表头

    Args:
        filename (str): 读取文件的文件名
        columns (List[str], optional): 选择表格数据希望保留的列. Defaults to None.
        sheetid (int, optional): 对于xlsx及xls文件，选择需要读取的表格是该文件的第几个表格. Defaults to 1.
        sep (str, optional): csv及txt文件的分割符. Defaults to ','.

    Returns:
        pd.DataFrame: 读取文件的表格数据，如果输入文件的文件名为csv, xlsx或xls
    或  List[str]: 读取txt文件的字符串列表
    
    Examples:
        >>> read_file('data.csv', ['col_1', 'col_2'])
        >>> read_file('data.xlsx', ['col_1', 'col_2', 'col_3'], sheetid=2)
        >>> read_file('sentences.txt', sep='\n')
    """    

    filename = filename.replace('\\', '/')
    filetype = filename.split('.')[-1]
    data = None
    if filetype == 'csv':
        data = pd.read_csv(filename, sep=sep)
    elif filetype == 'xlsx':
        buffer = StringIO()
        Xlsx2csv(filename, outputencoding='utf-8').convert(buffer, sheetid)
        buffer.seek(0)
        data = pd.read_csv(buffer)
    elif filetype == 'xls':
        data = pd.read_excel(filename)
    elif filetype == 'txt':
        f = open(filename, 'r', encoding='utf-8-sig')
        data = f.read().split(sep)
        f.close()

    if columns is not None and filetype in ('csv', 'xlsx', 'xls'):
        data = data[columns]
    return data

def save_file(data: pd.DataFrame, destination: str, filetype='csv') -> None:
    """保存表到指定路径

    Args:
        data (pd.DataFrame): 需要被保存的表
        destination (str): 保存的路径
        filetype (str, optional): 保存文件的类型. Defaults to 'csv'.
        
    Examples:
        >>> save_file(df1, 'df1.csv')
        >>> save_file(df2, 'df2.csv', filetype='csv')
    """    
    if filetype == 'csv':
        data.to_csv(destination, index=False, encoding='utf-8-sig')

def merge(df1: pd.DataFrame, df2: pd.DataFrame, left_on: List[str], right_on: List[str], drop_duplaicates=True) -> pd.DataFrame:
    """按照指定表头名称合并两个DataFrame

    Args:
        df1 (pd.DataFrame): 需要合并的第一个表
        df2 (pd.DataFrame): 需要合并的第二个表
        left_on (List[str]): 第一个表用来合并的表头名称.
        right_on (List[str]): 第二个表用来合并的表头名称.
        drop_duplaicates (bool, optional): 是否需要去掉重复的行. Defaults to True.

    Returns:
        pd.DataFrame: 按照要求合并后的表
        
    Examples:
        >>> merge(df1, df2, left_on=['col_1', 'col_2'], right_on=['col_2', 'col_3'], drop_duplicates=False)
    """
    df_merge = df1.merge(df2, 'left', left_on=left_on, right_on=right_on)
    df_merge.drop_duplicates(inplace=drop_duplaicates)
    return df_merge

def sift(data: pd.DataFrame, col_name: str, tgt_list: List[str]) -> pd.DataFrame:
    """筛选一个列中指定信息的行

    Args:
        data (pd.DataFrame): 需要被筛选的表
        col_name (str): 被选择列的表头名称
        tgt_list (List[str]): 筛选指定信息的列表

    Returns:
        pd.DataFrame: 信息筛选后的表
        
    Examples:
        >>> sift(df1, 'col_1', ['A', 'B'])
    """    
    return data.query(f'{col_name} in @tgt_list')

def drop_columns(data, cols):
    return data.drop(cols, 1)

# 分类数据变哑变量
def cat2ohe(X, cat):
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoder_df = pd.DataFrame(encoder.fit_transform(X[[cat]]).toarray(), columns=[f'{cat}_{i}' for i in sorted(X[cat].unique())]).astype('int')
    return encoder_df

# 含有分隔符的分类数据变哑变量
def cat2ohe_split(data_, id_col, val_col, delimiter='+'):
    data = data_[[id_col, val_col]]
    columns = []
    
    for d in data[val_col].unique(): columns += str(d).split(delimiter)
    columns = list(set(list(filter(None, columns))))
    results = pd.DataFrame('', index=data[id_col].unique(), columns=columns).astype(object)
    for _, row in tqdm(data.iterrows(), total=data.shape[0]):
        cols = row[1].split(delimiter)
        cols = list(set(list(filter(None, cols))))
        for d in cols: results[d][row[0]] = '1'
    results.index.name = id_col
    results = results.reset_index()
    return results

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

def similariy_prediction(inputs_, corpus_, threshold=0.7):
    try:
        from similarities import Similarity
    except:
        os.system('pip install similarities')
        from similarities import Similarity
        
    inputs, corpus = inputs_.copy(), corpus_.copy()
    try:
        inputs = inputs.tolist()
        corpus = corpus.tolist()
    except: pass
    inputs = [input.split(' ')[0] for input in inputs]
    def get_most_similar(sentences, corpus, topn):
        sentences = [re.sub("[\(\[（].*?[\)\]）]", "", s).split(' ')[0] for s in sentences]
        corpus = corpus
        model = Similarity(model_name_or_path="shibing624/text2vec-base-chinese")
        model.add_corpus(corpus)
        return model.corpus, model.most_similar(queries=sentences, topn=topn)
        
    corpus_dict, res = get_most_similar(inputs, corpus, 3)
    pred_list = []
    scores = []

    for i, c in tqdm(res.items()):
        c = {k: v for k, v in sorted(c.items(), key=lambda item: -item[1])}
        p = []
        max_s = -1
        for corpus_id, s in c.items():
            max_s = max(max_s, s)
            p.append(corpus_dict[corpus_id])

        pred_list.append(p)
        scores.append(max_s)

    pred_list = np.array(pred_list)
    out = pd.DataFrame({'inputs': inputs_, 'predictions': pred_list[:, 0], 'score': np.round(scores, 2)})
    out = out.sort_values('score', ascending=False)
    out['predictions'][out['score'] < threshold] = ''
    out['score'][out['score'] < threshold] = np.nan
    # out.to_csv('results.csv', index=False, encoding='utf-8-sig')
    return out