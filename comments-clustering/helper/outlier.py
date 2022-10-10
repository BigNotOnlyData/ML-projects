import pandas as pd

def remove_outlier_IQR(df: pd.DataFrame, column: str):
    """
    Удаляет выбросы
    df: таблица
    column: имя колонки, где удаляются выбросы
    """
    q25 = df[column].quantile(.25)
    q75 = df[column].quantile(.75)
    IQR = q75 - q25
    df_clean = df[~((df[column] > q75 + 1.5*IQR) | (df[column] < q25 - 1.5*IQR))]
    count_outlier = df.shape[0] - df_clean.shape[0]
    return df_clean, count_outlier

def remove_all_outlier_IQR(df: pd.DataFrame, column: str):
    """
    Удаляет выбросы, пока их не останется или будет достигнут предел итераций (100)
    df: таблица
    column: имя колонки, где удаляются выбросы
    """
    df_clean, count_outlier = remove_outlier_IQR(df, column)
    count = 0
    while count_outlier and count < 100:
        df_clean, count_outlier = remove_outlier_IQR(df_clean, column)
        count += 1
    print('Итераций удаления выбросов:', count)    
    return df_clean 


def remove_outlier_sigma(df: pd.DataFrame, column: str, sigma: int = 3):
    df_clean = df.copy()
    df_clean['std'] = (df[column] - df[column].mean()) / df[column].std()
    return df_clean[df_clean['std'].abs() <= sigma]
    