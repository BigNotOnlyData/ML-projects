
import pathlib
import pandas as pd
import numpy as np
import pickle


MODEL_FILE = pathlib.Path(__file__).parent.joinpath("my_model.pkl")
ENCODER_FILE = pathlib.Path(__file__).parent.joinpath('encoder.ohe')
AGG_COLS = ["material_code", "company_code", "country", "region", "manager_code"]
MATERIALS = ['material_lvl1_name', 'material_lvl2_name']
CAT_COLS = ['Contract + Spot',
             'class_company',
             'class_country',
             'quarter',
             'class_manager',
             'class_material',
             'material_lvl2_name',
             'month',
             'Контракт',
             'material_lvl1_name',
             'Спот',
             'class_region'
           ]



def transorm_to_ohe(df):
    """ Трансформирует тестовые данные в OHE """
    
    with open(ENCODER_FILE, 'rb') as f:
        encoder = pickle.load(f)
        
    df = df.copy()
    X_ohe = encoder.transform(df[CAT_COLS]).toarray()
    df_cat = pd.DataFrame(X_ohe, columns=encoder.get_feature_names(CAT_COLS), index=df.index)
    df_res = df.drop(CAT_COLS, axis=1).join(df_cat)
    return df_res


def get_autocorrelation(df):
    """Вычисляет значение тагрета следующего за таргетом с наибольшей автокорреляцией"""
    df = df.copy()
    # Создаем таблицу автокорреляций
    df_acorr = pd.DataFrame([], index=df.index)
    for lag in range(1, len(df.columns)):
        df_acorr[lag] = df.apply(lambda s: s.autocorr(lag=lag), axis=1).fillna(0) 
        
    # выбираем лаг для максимальной автокорреляции. 
    #(минус ставим для удобства поиска следущего значения, используя отрицательную индексацию)
    ind_max = -df_acorr.idxmax(axis=1) 
    
    best_target = []
    for seria, ind in zip(df.iterrows(), ind_max):
        best_target.append(seria[1].iloc[ind])
        
    seria_accor = pd.Series(best_target, name='target_next_max_accor')
    return seria_accor

    
def get_features(df: pd.DataFrame, month: pd.Timestamp) -> pd.DataFrame:
    """Вычисляет внутригрупповые признаки для месяца `month`."""
        
    # определяем начальную и конечную дату аггрегирования данных
    # смещением на 6 и 1 месяцев относительно предсказываемого месяца
    start_period = month - pd.offsets.MonthBegin(6)
    end_period = month - pd.offsets.MonthBegin(1)
    month_12 = month - pd.offsets.MonthBegin(12)
    ##### Годовой интервал #####
    df_year = df.loc[:, month_12:end_period].copy()
    
    # выбираем даты по end_period включительно
    df = df.loc[:, :end_period]
    
    features = pd.DataFrame([], index=df.index)
    # создаем признак месяца
    features["month"] = month.month
    # квартал
    features["quarter"] = month.quarter
    # создаем 6 призаков - 6 последних месяцев
    features[[f"vol_tm{i}" for i in range(6, 0, -1)]] = df.loc[:, start_period:end_period].copy()
    
    # скользящие значения с окном 12 месяцев
    rolling = df.rolling(12, axis=1, min_periods=1)
    # среднее
    features = features.join(rolling.mean().iloc[:, -1].rename("last_year_avg"))
    # минимум
    features = features.join(rolling.min().iloc[:, -1].rename("last_year_min"))
    # максимум
    features = features.join(rolling.max().iloc[:, -1].rename("last_year_max"))
    
    # скользящие значения с окном 6 месяцев
    rolling = df.rolling(6, axis=1, min_periods=1)
    # среднее
    features = features.join(rolling.mean().iloc[:, -1].rename("half_year_avg"))
    # минимум
    features = features.join(rolling.min().iloc[:, -1].rename("half_year_min"))
    # максимум
    features = features.join(rolling.max().iloc[:, -1].rename("half_year_max"))

    
    # экспотенциальное сглаживание
    ewm = df.ewm(alpha=0.5, adjust=False, axis=1)
    ewm_mean = ewm.mean().iloc[:, -1].rename("ewm_0.5")
    features = features.join(ewm_mean)
    
    # отношение разницы лагов от 1 до 6 к первому лагу 
    for lag in range(1, 7):
        diff = df.diff(periods=lag, axis=1)
        features[f'diff_ratio_{lag}'] =  diff.iloc[:, -1] / (features['vol_tm1'] + 1e-3)
    
    # относительные годовые статистики
    features['last_year_max_to_1'] = features['vol_tm1'] / (features["last_year_max"] + 1e-3)
    features['last_year_min_to_1'] = features['vol_tm1'] / (features["last_year_min"] + 1e-3)
    features['last_year_avg_to_1'] = features['vol_tm1'] / (features["last_year_avg"] + 1e-3)
    
    ### Автокорреляция ###
    seria_accor = get_autocorrelation(df)
    features = features.join(seria_accor)
    
    return features

def get_features2(df_general, month):
    """  Внешнегрупповые признаки:
            1. Вычисляются средние показатели за предыдущий месяц для каждой компоненты группы
            2. Каждая компонента группы разбивается на классы     
    """
    month_ago = month - pd.offsets.MonthBegin(1)
    month_start = month - pd.offsets.MonthBegin(12)
    month_period = pd.date_range(month_start, month_ago, freq='MS')

    df = df_general.copy()
    features = df[AGG_COLS + MATERIALS]
    #среднее в стране за последний месяц
    df_country = df.groupby('country', as_index=False)[[month_ago]]\
                           .mean()\
                           .rename({month_ago:'country_mean_month_ago'}, axis=1)
    features = features.merge(df_country, how='left')
    
    #среднее по региону за последний месяц
    df_region = df.groupby('region', as_index=False)[[month_ago]]\
                           .mean()\
                           .rename({month_ago:'region_mean_month_ago'}, axis=1)
    
    features = features.merge(df_region, how='left')
   
    #среднее по коду компании за последний месяц
    df_company = df.groupby('company_code', as_index=False)[[month_ago]]\
                           .mean()\
                           .rename({month_ago:'company_mean_month_ago'}, axis=1)
    features = features.merge(df_company, how='left')
    
    #среднее по менеджеру за последний месяц
    df_manager = df.groupby('manager_code', as_index=False)[[month_ago]]\
                           .mean()\
                           .rename({month_ago:'manager_mean_month_ago'}, axis=1)  
    features = features.merge(df_manager, how='left')
    
    #среднее по коду товара за последний месяц
    df_material_code = df.groupby('material_code', as_index=False)[[month_ago]]\
                           .mean()\
                           .rename({month_ago:'material_mean_month_ago'}, axis=1) 
    features = features.merge(df_material_code, how='left')
    
    # Сочетание всех компонент группы
    features['mix_ewm'] = 0.2 * features['material_mean_month_ago'] + 0.2 * features['manager_mean_month_ago'] \
                          + 0.2 * features['region_mean_month_ago'] + 0.2 * features['company_mean_month_ago'] \
                          + 0.2 * features['country_mean_month_ago']
    
    ################## КЛАССЫ ###################
    # разбиение менеджеров на классы по количесту групп
    mclass = df['manager_code'].value_counts()\
                               .apply(lambda x: 0 if x == 1 else
                                      (1 if x<10 else 
                                       (2 if x <=40 else 3)))\
                               .rename('class_manager')
    features = features.merge(mclass.to_frame(), right_index=True, left_on='manager_code', how='left')
    
    # разбиение компаний на классы по количесту групп
    company_class = df['company_code'].value_counts()\
                               .apply(lambda x: 0 if x == 1 else
                                      (1 if x<10 else 
                                       (2 if x <=40 else 3)))\
                               .rename('class_company')
    features = features.merge(company_class.to_frame(), right_index=True, left_on='company_code', how='left')
    
    # разбиение продукта на классы по количесту групп
    material_class = df['material_code'].value_counts()\
                               .apply(lambda x: 0 if x == 1 else
                                      (1 if x<10 else 
                                       (2 if x <=40 else 3)))\
                               .rename('class_material')
    features = features.merge(material_class.to_frame(), right_index=True, left_on='material_code', how='left')
    
    # разбиение регионов на классы по количесту групп
    region_class = df['region'].value_counts()\
                               .apply(lambda x: 0 if x == 1 else
                                      (1 if x<10 else 
                                       (2 if x <20 else 
                                        (3 if x <45 else 4))))\
                               .rename('class_region')
    features = features.merge(region_class.to_frame(), right_index=True, left_on='region', how='left')
    
    # Разбиение стран на класссы
    Asia = ['Китай', 'Индия', 'Турция']
    Europe = ['Испания', 'Эстония', 'Дания', 'Словакия', 'Швеция', 'Венгрия', 'Швейцария', 'Литва', 'Соед. Королев.',
         'Бельгия', 'Сербия', 'Нидерланды', 'Италия', 'Австрия', 'Финляндия', 'Франция', 'Чехия', 'Германия',
         'Польша']
    
    country_class = df['country'].apply(lambda x: 0 if x == 'Россия' 
                                            else (1 if x in Asia 
                                                  else (2 if x in Europe else 3)))\
                                 .rename('class_country')
    features['class_country'] = country_class
    return features

def get_features_contract(df, month):
    """ 1. Вычисляем общее количество сделок в прошлом месяце, за полгода и частоту сделок 
        2. Ставим флаги для контрактов каждого типа в прошлом месяце
    """
    month_ago = month - pd.offsets.MonthBegin(1)
    month_6_ago = month - pd.offsets.MonthBegin(6)
    period = pd.date_range(month_6_ago, month_ago, freq='MS')
    
    df = df.copy()
    # количество сделок за прошлый месяц
    df_contract = df.loc[:, (slice(None), [month_ago])]
    features = df_contract.sum(axis=1).to_frame().rename({0:'count_deal_mounth_ago'}, axis=1)
    
    # количество сделок за полгода
    df_contract_6m = df.loc[:, (slice(None), period)]
    features_6m = df_contract_6m.sum(axis=1).rename('count_deal_period_6_mounth')
    
    # частота сделок за полгода
    frequency_deal_half_year = features_6m / len(period)
    
    # отношение частот сделок
    features['ratio_deal_half_to_1'] = features['count_deal_mounth_ago'] /\
                                     (frequency_deal_half_year + 1e-3)
    
    # Флаги для наименования сделки в прошлом месяце
    df_deal = df_contract.where(df_contract == 0, 1).droplevel(1, axis=1)
    features = features.join(df_deal)
    return features.reset_index()

def get_days_last_contact(df, month):
    """ Вычисляем какое количество дней назад была совершена последняя сделка, относительно месяца 'month',
        ограничиваясь интервалом в 12 месяцев, и в конце берем обратную величину от найденного количества.
    """
    df = df.copy()
    # таблица с количеством дней от последней сделки, но не более чем 12 месяцев
    df_days = pd.DataFrame(np.zeros_like(df.index), index=df.index, columns=['last_deal_in_days'])
    for lag in range(1, 13):
        # прошлый месяц
        last_month = month - pd.offsets.MonthBegin(lag)
        # всего дней в месяце
        all_days = last_month.days_in_month
        # день последней сделки, 0 - если не было сделок
        df_lag = df[last_month].apply(lambda row: row[row>0].index.max().day, axis=1).fillna(0)
        # разница в днях от предсказываемого месяца до последней сделки
        df_diff = (all_days - df_lag).rename(f'lag_{lag}')
        # присоединяем к заготовке
        df_days = df_days.join(df_diff)
        # отбираем только те группы в которых сделок в текущем месяце не было 
        df = df.loc[df_diff[df_diff==all_days].index]
    df_days['last_deal_in_days'] = df_days.sum(axis=1) 
    # берем обратную величину
    df_days['ratio_last_deal'] = 1 / (df_days['last_deal_in_days'] + 1)
    return df_days.loc[:,'ratio_last_deal'].reset_index()


def get_all_features(data, month):
    # сформируем вспомогательные таблицы
    df_general = data.groupby(AGG_COLS + MATERIALS + ["month"])['volume'].sum().unstack(fill_value=0).reset_index()
    df_target = df_general.select_dtypes('float')
    df_contract = data.groupby(AGG_COLS+['contract_type']+['month']).size()\
                  .unstack(['contract_type']+['month'], fill_value=0)\
    
    df_features = df_general[AGG_COLS + MATERIALS]
    df_date = data.groupby(AGG_COLS+['month']+['date'])['volume'].sum().unstack(['month', 'date'], fill_value=0)
    
    # логарифмирование ЦП
    df_target_log = np.log1p(df_target)
    
    # создадим признаки
    df_f1 = get_features(df_target_log, month)
    month_features = df_features.join(df_f1)
    
    df_f2 = get_features2(df_general, month)
    month_features = month_features.merge(df_f2, how='left')

    df_f3 = get_features_contract(df_contract, month)
    month_features = month_features.merge(df_f3, how='left')
    
    df_f4 = get_days_last_contact(df_date, month)
    month_features = month_features.merge(df_f4, how='left')
    
    # удалим ненужные колонки
    drop_cols = ['company_code', 'region', 'country', 'material_code', 'manager_code',]
    month_features = month_features.drop(drop_cols, axis=1)
    
    # преобразуем катеориальные фичи в OHE
    month_features = transorm_to_ohe(month_features)
    return month_features


def predict(df: pd.DataFrame, month: pd.Timestamp) -> pd.DataFrame:
    """
    Вычисление предсказаний.

    Параметры:
        df:
          датафрейм, содержащий все сделки с начала тренировочного периода до `month`; типы
          колонок совпадают с типами в ноутбуке `[SC2021] Baseline`,
        month:
          месяц, для которого вычисляются предсказания.

    Результат:
        Датафрейм предсказаний для каждой группы, содержащий колонки:
            - `material_code`, `company_code`, `country`, `region`, `manager_code`,
            - `prediction`.
        Предсказанные значения находятся в колонке `prediction`.
    """
    features = get_all_features(df, month)
    
    with open(MODEL_FILE, 'rb') as f2:
        model = pickle.load(f2)
#     model = CatBoostRegressor()
#     model.load_model(MODEL_FILE)

    predictions = model.predict(features).clip(.0)
    norm_pred = np.expm1(predictions)
    
    df_res = df.groupby(AGG_COLS + ["month"])['volume'].sum().unstack(fill_value=0).reset_index()
    preds_df = df_res[AGG_COLS].copy()
    preds_df["prediction"] = np.floor(norm_pred)

    return preds_df
