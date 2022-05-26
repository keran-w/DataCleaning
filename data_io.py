from utils import *


def read_file(filename: str, columns=None, sheetid=1, sep=',', use_fast=False) -> Union[pd.DataFrame, List[str]]:
    """ 读取多种文件类型的数据, 对于列表数据, 可以选择部分表头

    Args:
        filename (str): 读取文件的文件名
        columns (List[str], optional): 选择表格数据希望保留的列. Defaults to None.
        sheetid (int, optional): 对于xlsx及xls文件, 选择需要读取的表格是该文件的第几个表格. Defaults to 1.
        sep (str, optional): csv及txt文件的分割符. Defaults to ','.
        use_fast (bool, optional): 是否首先将xlsx文件转为csv文件, 这样会对xlsx文件的读取提速. Defaults to False.

    Returns:
        Union[pd.DataFrame, List[str]]: 
            读取文件的表格数据, 如果输入文件的文件名为csv, xlsx或xls
            或读取txt文件的字符串列表

    Examples:
        >>> read_file('data.csv', ['col_1', 'col_2'])
        >>> read_file('data.xlsx', ['col_1', 'col_2', 'col_3'], sheetid=2)
        >>> read_file('sentences.txt', sep='\n')
    """

    filename = filename.replace('\\', '/')
    filetype = filename.split('.')[-1]
    data = None
    from io import StringIO
    if filetype == 'csv':
        data = pd.read_csv(filename, sep=sep)
    elif filetype == 'xlsx':
        if use_fast:
            from xlsx2csv import Xlsx2csv
            buffer = StringIO()
            Xlsx2csv(filename, outputencoding='utf-8').convert(buffer, sheetid)
            buffer.seek(0)
            data = pd.read_csv(buffer)
        else:
            data = pd.read_excel(filename, sheetid)

    elif filetype == 'xls':
        data = pd.read_excel(filename, sheetid)
    elif filetype == 'txt':
        f = open(filename, 'r', encoding='utf-8-sig')
        data = f.read().split(sep)
        f.close()
    if data is None:
        raise f'filetype <{filetype}> is not supported'

    if columns is not None and filetype in ('csv', 'xlsx', 'xls'):
        data = data[columns]
    return data


def save_file(data: Union[pd.DataFrame, List], destination: str, filetype: str = None) -> None:
    """保存表到指定路径

    Args:
        data (pd.DataFrame): 需要被保存的表
        destination (str): 保存的路径
        filetype (str, optional): 保存文件的类型. Defaults to None.

    Examples:
        >>> save_file(df1, 'df1.csv')
        >>> save_file(df2, 'df2.csv', filetype='csv')
    """

    if filetype is None:
        filetype = destination.split('.')[-1]
    elif destination.split('.')[-1] in ('csv', 'txt'):
        pass
    else:
        destination += f'.{filetype}'

    if filetype == 'csv':
        data.to_csv(destination, index=False, encoding='utf-8-sig')
    elif filetype == 'txt':
        f = open(destination, 'w', encoding='utf-8')
        f.write('\n'.join([f'{item}' for item in data]))
        f.close()
