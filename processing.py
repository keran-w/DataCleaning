from utils import *


def base_table_process(data_csv: pd.DataFrame, id: str, name: str, key: str, values: Union[str, List[str]]) -> pd.DataFrame:
    """建立基础表

    Args:
        data_csv (pd.DataFrame): 包含所有用来建立基础表信息的DataFrame
        id (str): 唯一区分每行的id
        name (str): 基础表基础信息列, 该列的不重复信息将位于每条信息的第一列
        key (str): 基础表基础信息列的填充值
        values (Union[str, List[str]]): 基础表其他信息列的表头名

    Returns:
        pd.DataFrame: 返回建立的基础表

    Examples:
        >>> results = base_table_process(data, id, name, key, values)
        >>> results.index.name = id
        >>> results = results.reset_index().fillna('')
        >>> results.to_csv(output_filename, index=False, encoding='utf-8-sig')
    """

    data_csv = data_csv.query(f'{name} == {name}')[
        [id, name, key] + values].drop_duplicates()
    data_csv[id] = data_csv[id].astype('string')
    num_values = len(values)
    new_indices = []
    for _, (i, count) in data_csv.groupby([id, name], sort=False).count().max(level=0).max(1).reset_index().iterrows():
        new_indices += [i] * count

    value_count = len(values) + 1
    new_columns = data_csv[name].unique().repeat(value_count)
    for i in range(1, value_count):
        new_columns[i::value_count] += f'_{values[i - 1]}'

    results = pd.DataFrame('', index=new_indices,
                           columns=new_columns).astype(object)
    first_index = {id: row_num for _, (row_num, id) in
                   pd.DataFrame(results.index).drop_duplicates().reset_index().iterrows()}
    index_count = {nameid: 0 for nameid in (
        data_csv[name] + data_csv[id]).unique()}

    def base_table_process_helper(results, row, values, num_values, first_index, index_count):
        i, n, v = row[0], row[1], row[2:]
        flag = 1
        try:
            results[n][i][0]
        except:
            flag = 0

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

    from tqdm import tqdm
    for _, row in tqdm(data_csv[[id, name, key] + values].iterrows(), total=data_csv.shape[0]):
        base_table_process_helper(results, row, values,
                                  num_values, first_index, index_count)

    return results


def similariy_prediction(inputs: List[str], corpus: List[str], top_n=2, threshold=0.8, sort_score=False) -> pd.DataFrame:
    """预测一个输入字符串的归一化结果

    Args:
        inputs (List[str]): 待归一化的输入字符串列表
        corpus (List[str]): 匹配归一化结果的语料库
        top_n (int, optional): 返回归一化分数大于阈值的个数. Defaults to 2.
        threshold (float, optional): 归一化分数的阈值, 该阈值越大, 则所预测的可信度越高, 范围: 0~1. Defaults to 0.8.
        sort_score (bool, optional): 是否将最高的归一化分数排序. Defaults to False.

    Returns:
        pd.DataFrame: 返回包含原输入信息、归一化预测和预测分数的DataFrame

    Examples:
        >>> inputs = read_file('inputs.txt, sep='\n')
        >>> corpus = read_file('corpus.txt, sep='\n')
        >>> sim_results = similariy_prediction(inputs, corpus, top_n=3, threshold=0.75, sort_score=True)
        >>> save_file(sim_results, 'sim_results.csv')
    """

    def get_cosine_similarities(inputs: List[str], corpus: List[str]):
        from text2vec import SentenceModel
        from sklearn.preprocessing import normalize
        max_seq_length = (max([len(i) for i in inputs] +
                              [len(c)for c in corpus]) // 64 + 1) * 64
        model = SentenceModel(max_seq_length=max_seq_length)
        similarities = normalize(model.encode(inputs)) \
            @ normalize(model.encode(corpus)).T
        return similarities

    sim = get_cosine_similarities(inputs, corpus)
    res = pd.DataFrame({'输入': inputs})
    for k in range(top_n):
        res[f'预测{k+1}'] = [corpus[i] if sim[j, i] > threshold else ''
                           for j, i in enumerate(np.argsort(sim, 1)[:, ::-1][:, k])]
        res[f'预测{k+1}分数'] = [i if i > threshold else 0
                             for i in np.sort(sim, 1)[:, ::-1][:, k]]
    if sort_score:
        return res.sort_values('预测1分数', ascending=False).replace(0, '')
    else:
        return res.replace(0, '')


def extract_dose_and_unit_from_text(
    inputs_: List[str],
    units: List[str],
    il_units: List[str] = [],
    replace_dict: Dict[str, str] = {},
    ENC: str = 'A'
):
    """_summary_

    Args:
        inputs_ (List[str]): _description_
        units (List[str]): _description_
        il_units (List[str], optional): _description_. Defaults to [].
        replace_dict (Dict[str, str], optional): _description_. Defaults to {}.
        ENC (str, optional): _description_. Defaults to 'A'.

    Returns:
        _type_: _description_
    """
    import re
    import jieba
    from itertools import chain
    from pypinyin import lazy_pinyin as pinyin
    import pandas as pd
    import numpy as np

    inputs = [' '.join(item.split()).lower() for item in inputs_]
    replace_dict['.'] = ENC

    def replace_by_dict(s):
        for k, v in replace_dict.items():
            s = s.replace(k, v)
        return s

    inputs = [replace_by_dict(item) for item in inputs]

    encoding_dict = {}
    for unit in units + il_units:
        unit_encoding = ''.join(list(chain.from_iterable(pinyin(unit))))[:4]
        encoding_dict[unit_encoding] = unit
        inputs = [re.sub(f'{unit}| {unit}', unit_encoding + ' ', item)
                  for item in inputs]
    inputs = [' '.join(item.split()) for item in inputs]
    outputs = [jieba.lcut(item, cut_all=False) for item in inputs]

    def get_unit(item, return_unit=True):
        flag = ENC in item
        item_ = item.replace(ENC, '')
        match = re.match(r"([0-9]+)([a-z]+)", item_, re.I)
        if match:
            if return_unit:
                return match.groups()[-1]
            else:
                if flag:
                    return item[:-len(match.groups()[-1])].replace(ENC, '.'), match.groups()[-1]
                else:
                    return match.groups()
        else:
            if return_unit:
                return ''
            else:
                return ('', '')

    def filter_item(item):
        res = list(filter(lambda x: x != '', item))
        try:
            return res[0]
        except:
            return ''

    outputs = [[item if get_unit(item) in list(
        encoding_dict.keys()) else '' for item in items] for items in outputs]
    outputs = [filter_item(item) for item in outputs]
    outputs = [get_unit(item, return_unit=False) for item in outputs]
    outputs = np.array(outputs)

    res = pd.DataFrame({'inputs': inputs_})
    res['outputs'] = outputs[:, 0]
    res['unit'] = outputs[:, 1]
    encoding_dict[''] = ''
    res['outputs'] = res['outputs']
    res['unit'] = res['unit'].apply(lambda x: encoding_dict[x])
    res['unit'] = res['unit'].apply(lambda x: '' if x in il_units else x)
    res['outputs'] = res['outputs'].apply(lambda x: x.replace(ENC, '.'))
    return res
