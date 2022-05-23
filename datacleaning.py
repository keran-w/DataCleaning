from utils import *


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


def extract_unique(df: pd.DataFrame, col: str) -> List[str]:
    """读取DataDFrame中特定列的唯一数据

    Args:
        df (pd.DataFrame): 需要进行相似性预测的列
        col (str): 相似性预测的目标列

    Returns:
        List[str]: 唯一字符串列表
    """

    return df[col].unique().tolist()


def drop_columns(data: pd.DataFrame, cols: Union[str, List[str]], verbose=False) -> pd.DataFrame:
    """从DataFrame中删除一列或多列数据(将只删除存在的列)

    Args:
        data (pd.DataFrame): 将要删除的DataFrame
        cols (Union[str, List[str]]): 将要删除的列名
        verbose (bool, optional): _description_. Defaults to False.

    Returns:
        pd.DataFrame: 删除后的DataFrame
    """
    
    data_ = data.copy()
    except_cols = []
    for col in cols:
        try:
            data_ = data_.drop(col, 1)
        except:
            except_cols.append(col)

    if verbose:
        print('Removing columns: {}'.format(
            ''.join([f'{col} ' if col not in except_cols else '' for col in cols])))
        print('Cannot remove columns: {}'.format(' '.join(except_cols)))
    return data_


def cat2ohe(data: pd.DataFrame, cat: str) -> pd.DataFrame:
    """分类数据变哑变量

    Args:
        data (pd.DataFrame): _description_
        cat (str): _description_

    Returns:
        pd.DataFrame: _description_
    """
    
    from sklearn.preprocessing import OneHotEncoder
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoder_df = pd.DataFrame(encoder.fit_transform(data[[cat]]).toarray(), columns=[
                              f'{cat}_{i}' for i in sorted(data[cat].unique())]).astype('int')
    return encoder_df


def cats2ohe(data: pd.DataFrame, cats: List[str]) -> pd.DataFrame:
    """_summary_

    Args:
        data (pd.DataFrame): _description_
        cats (List[str]): _description_

    Returns:
        pd.DataFrame: _description_
    """
    
    from sklearn.preprocessing import OneHotEncoder
    results = pd.DataFrame(index=data.index)
    for cat in cats:
        encoder = OneHotEncoder(handle_unknown='ignore')
        encoder_df = pd.DataFrame(encoder.fit_transform(data[[cat]]).toarray(), columns=[
                                  f'{cat}_{i}' for i in sorted(data[cat].unique())]).astype('int')
        results = pd.concat([results, encoder_df], axis=1)
    return results


def cat2ohe_split(data_: pd.DataFrame, id_col: str, val_col: str, delimiter='+') -> pd.DataFrame:
    """含有分隔符的分类数据变哑变量

    Args:
        data_ (pd.DataFrame): _description_
        id_col (str): _description_
        val_col (str): _description_
        delimiter (str, optional): _description_. Defaults to '+'.

    Returns:
        pd.DataFrame: _description_
    """
    
    from tqdm import tqdm
    data = data_[[id_col, val_col]]
    columns = []

    for d in data[val_col].unique():
        columns += str(d).split(delimiter)
    columns = list(set(list(filter(None, columns))))
    results = pd.DataFrame(
        '', index=data[id_col].unique(), columns=columns).astype(object)
    for _, row in tqdm(data.iterrows(), total=data.shape[0]):
        cols = row[1].split(delimiter)
        cols = list(set(list(filter(None, cols))))
        for d in cols:
            results[d][row[0]] = '1'
    results.index.name = id_col
    results = results.reset_index()
    return results


def rank_time(data_: pd.DataFrame, time_col: str, other_cols: List[str], ascending=True) -> pd.DataFrame:
    """_summary_

    Args:
        data_ (pd.DataFrame): _description_
        time_col (str): _description_
        other_cols (List[str]): _description_
        ascending (bool, optional): _description_. Defaults to True.

    Returns:
        pd.DataFrame: _description_
    """
    
    data = data_.copy()
    data[time_col] = pd.to_datetime(data[time_col])
    all_cols = other_cols + [time_col]
    data = data.sort_values(all_cols)
    data['rank'] = data[all_cols].groupby(other_cols, sort=False)[
        time_col].rank(ascending=ascending, method='first')
    return data


def remove_negative_cost(data: pd.DataFrame, time_col: str, cost_col: str, other_cols: List[str]) -> pd.DataFrame:
    """_summary_

    Args:
        data (pd.DataFrame): _description_
        time_col (str): _description_
        cost_col (str): _description_
        other_cols (List[str]): _description_

    Returns:
        pd.DataFrame: _description_
    """
    
    del_list = []
    prev_idx = 0
    for idx, row in data[data.duplicated(other_cols, keep=False)].sort_values(other_cols + [time_col]).iterrows():
        if row[cost_col] < 0:
            del_list += [prev_idx, idx]
        prev_idx = idx
    return data.drop(del_list)


def remove_empty_cells(data: pd.DataFrame, col: str) -> pd.DataFrame:
    """去除数据中在某一列中为空值的所有数据

    Args:
        data (pd.DataFrame): _description_
        col (str): _description_

    Returns:
        pd.DataFrame: _description_
    """
    
    return data.query(f'{col} == {col}')
