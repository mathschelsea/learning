import pandas as pd

def df_look(df):
    """Displays the number of rows and columns in the dataframe. Also lists
    each column name, it's data type and the number of unique levels it has.
    -----------
    df: A pandas dataframe.
    Examples:
    ---------
    >>> df = pd.DataFrame({'col1' : [1, 0, 1], 'col2' : ['a', 'b', 'c']})
    >>> df
       col1 col2
    0     1    a
    1     0    b
    2     1    c
    >>> df_look(df)
    Number of rows: 3
    Number of columns: 2

    Column name, type, number of unique values:
    col1  int64  2
    col2  object  3
    """
    print(f'Number of rows: {df.shape[0]}')
    print(f'Number of columns: {df.shape[1]}')
    print('')
    print('Column name, type, number of unique values:')
    for i in range(0, df.shape[1]):
        print(' ', df.columns[i], '', df.dtypes[i], '', df[df.columns[i]].nunique())