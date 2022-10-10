from typing import List, Union, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.metrics import (calinski_harabasz_score,
                             davies_bouldin_score,
                             silhouette_samples,
                             silhouette_score)
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm, trange
from yellowbrick.utils import KneeLocator
from wordcloud import WordCloud

NOISE = -1
NOISE_NAME = 'Шум'
NOISE_COLOR = '#777676'


def plot_explained_ratio(X: np.array, model: Union[TruncatedSVD, PCA], \
                         title: str, treshold: float, ax: Optional[plt.Axes] = None):
    """
    Строит график доли объясняющей дисперсии при снижение размерности
    X: данны
    model: модель снижения размерности
    title: имя графика
    ax: plt.Axes
    treshold: порог объясняющей доли
    """
    n_components = X.shape[1] - 1
    model.set_params(n_components=n_components).fit(X)
    
    if not ax:
        fig, ax = plt.subplots(figsize=(6, 4))

    expr = np.cumsum(model.explained_variance_ratio_)
    opt = np.where(expr > treshold)
    if opt:
        ax.axvline(opt[0][0]+1, c='g', ls='--', label='Оптимальное число компонент')
        title_str = title + f'. x={opt[0][0]}'

    ax.axhline(treshold, c='r', ls='--', label=f'Порог: y={treshold}')
    ax.plot(range(1, n_components+1), expr, label='explained_variance_ratio_')
    ax.legend(loc='lower right', fontsize=16)
    ax.set_xlabel('Количество компонент', fontsize=16)
    ax.set_ylabel('explained variance ratio', fontsize=16)
    ax.set_title(title_str, fontsize=18)
    ax.tick_params(axis='both', labelsize=14)
    ax.grid(False)
    
    
def multiplot_explained_ratio(model: Union[TruncatedSVD, PCA], X: List[np.array], title: List[str], treshold: float = .9) -> None:
    """
    Строит графики доли объясняющей дисперсии при снижение размерности
    model: модель снижения размерности
    X: список данных
    title: список соответсвующих имен графиков
    treshold: порог объясняющей доли
    """
    assert len(X) == len(title), 'X и title разной длины'
    rows = len(X) // 2  
    extra = False
    if len(X) % 2: 
        rows += 1
        extra = True
        
    fig, axes = plt.subplots(rows, 2, figsize=(18, 6*rows))
    axes = axes.flatten()
    
    if extra:
        axes = axes[:-1]
        
    for i, ax in enumerate(axes):
        plot_explained_ratio(X[i], model=model, title=title[i], ax=ax, treshold=treshold)  

    fig.suptitle(f"{str(model).split('(')[0]}", fontsize=22, fontweight="bold")
    fig.tight_layout() 
    
    
def plot_3d(df: pd.DataFrame,
            x='x',
            y='y',
            z='z',
            color: Optional[str] = None,
            color_discrete_map: Optional[dict] = None):
    """
    Создаёт трехмерную визуализацию
    :param df: таблица с данными с обязательными колонками x, y, z
    :param x: размерность x
    :param y: размерность y
    :param z: размерность z
    :param color: имя колонки кластеров
    :param color_discrete_map: словарь с цветами для кластеров
    :return: figure plotly
    """
    # если передан цвет (кластеры)
    if color is not None:
        # строковый тип нужен для дискретного отображения кластеров plotly
        df_ = df.copy()
        df_.loc[:, color] = df_[color].astype(str)
        # Разделяем на два графика чтобы прозрачность кластера Шума регулироовать, другое решение?
        fig_outlier = None
        df_outlier = df_[df_[color] == str(NOISE)]
        df_norm = df_[df_[color] != str(NOISE)]
        # отрисовываем выбросы если есть
        if df_outlier.shape[0]:
            fig_outlier = px.scatter_3d(df_outlier,
                                        x=x,
                                        y=y,
                                        z=z,
                                        opacity=.4,
                                        color_discrete_sequence=[NOISE_COLOR],
                                        hover_name=[NOISE_NAME] * len(df_outlier),
                                        )

        # отрисовываем кластеры
        fig = px.scatter_3d(df_norm,
                            x=x,
                            y=y,
                            z=z,
                            color=color,
                            labels={color: 'Кластер'},
                            color_discrete_map=color_discrete_map,
                            
                            )

        # если есть выбросы объединяем данные
        data = fig.data
        if fig_outlier is not None:
            data = data + fig_outlier.data

        fig = go.Figure(data=data)
    else:
        fig = px.scatter_3d(df, x=x, y=y, z=z)
        
    fig.update_layout(legend=dict(
    orientation="v", 
    yanchor="top",
    y=1,
    xanchor="right",
    x=1,
    title_text='Кластер',
))    

    # fig.update_layout(title_text='Проекция комментариев в 3-мерном пространстве', title_x=0.5)
    return fig 


def plot_silhouette(X: np.array, model, k=(2, 20)):
    """
    Отрисовка графиков силуэта
    X: данные для обучения
    model: модель кластеризации
    k: диапазон кластеров для анализа
    """
    n_min, n_max = k
    for n_clusters in range(n_min, n_max+1):
        fig, ax = plt.subplots(1, 2, figsize=(18, 6))
        # обучаем кластеризатор и считаем метрики
        model.set_params(n_clusters=n_clusters)
        model.fit(X)
        silhouette_avg = silhouette_score(X, model.labels_)
        sample_silhouette_values = silhouette_samples(X, model.labels_)

        # устанавливаем настройки графика силуэта
        ax[1].set_xlabel(f"Коэффициент силуэта", fontsize=16)
        ax[1].set_xlim([-0.2, 1])
        ax[1].set_ylim([0, len(X)])
        ax[1].axvline(x=silhouette_avg, color="red", linestyle="--", label=f"Средний коэф. силуэта = {silhouette_avg:.3f}")
        ax[1].legend(loc='lower right', fontsize=16)
        ax[1].set_yticks([])
        ax[1].set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        ax[1].tick_params(axis='both', labelsize=14)

        # устанавливаем настройки графика количества объектов
        ax[0].set_ylabel(f"Номер кластера", fontsize=16)
        ax[0].set_xlabel(f"Количество объектов", fontsize=16)
        ax[0].set_yticks(range(n_clusters))
        ax[0].tick_params(axis='both', labelsize=14)
        ax[0].grid(True, axis='x')
        ax[0].grid(False, axis='y')

        # Рисуем графики для каждого кластера
        y_lower = 0
        for n in range(n_clusters):
            # задаем уникальный цвет из палитры
            color = plt.cm.nipy_spectral(n / n_clusters)

            # отбираем метрику для конкретного кластера
            cluster_silhouette_values = sample_silhouette_values[model.labels_ == n]
            cluster_silhouette_values.sort()
            size_cluster = cluster_silhouette_values.shape[0]

            # закрашиваем область
            y_upper = y_lower + size_cluster
            ax[1].fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                cluster_silhouette_values,
                alpha=0.7,
                color=color
            )

            # обозначаем номер кластера
            ax[1].text(-0.05, y_lower + 0.5 * size_cluster, str(n), fontsize=14)

            # рисуем столбик
            ax[0].barh(n, size_cluster, color=color)

            y_lower = y_upper

        fig.suptitle(f"Количество кластеров: {n_clusters}", fontsize=20, fontweight="bold") 
        fig.tight_layout()    
        plt.show()

def cluster_metrics(X, model, k=(2, 20), plot=True, best_cluster=True):
    """
    Вычисляет метрики, строит графики и находит лучшее число кластеров
    X: данные для обучения
    model: модель кластеризации
    k: диапазон кластеров для анализа
    plot: рисовать или нет графики
    best_cluster: считать или нет лучшее число кластеров
    """
    n_min, n_max = k
    # считаем метрики
    calinski_harabasz_scores = []
    davies_bouldin_scores = []
    silhouette_scores = []
    for n in trange(n_min, n_max+1):
        model.set_params(n_clusters=n)
        model.fit(X)
        calinski_harabasz_scores.append(calinski_harabasz_score(X, model.labels_))
        davies_bouldin_scores.append(davies_bouldin_score(X, model.labels_))
        silhouette_scores.append(silhouette_score(X, model.labels_))

    # рисуем графики
    if plot:
        scores = [calinski_harabasz_scores, davies_bouldin_scores, silhouette_scores]
        names = ['Calinski-Harabasz Index', 'Davies-Bouldin Index', 'Silhouette Coefficient']
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        for ax, score, name in zip(axes, scores, names):      
            ax.plot(range(n_min, n_max+1), score, 'bo-')
            ax.set_title(f'{name}', fontsize=18)
            ax.set_xticks(range(n_min, n_max+1, 3))
            ax.grid(True)
            ax.tick_params(axis='both', labelsize=12)

        fig.supxlabel('Количество кластеров', fontsize=14)
        plt.show()
        
    # вычисляем лучшее количество кластеров    
    if best_cluster:
        df_metrics = pd.DataFrame(data={'cal_har': calinski_harabasz_scores, 
                                        'dav_bould': davies_bouldin_scores, 
                                        'silhouette': silhouette_scores
                                       }, 
                                  index=range(n_min, n_max+1)
                                 )
        
        # инвертируем для удобства сравнивания метрик по максимуму 
        # (так как у этой метрики лучшие значения у минимума)
        df_metrics['dav_bould'] = 1 / df_metrics['dav_bould']
        
        # Приводим к одному маштабу для дальнейшего объединения метрик
        scaler = MinMaxScaler()
        df_metrics[df_metrics.columns] = scaler.fit_transform(df_metrics)

        # объединенная метрика
        union_metric = 1/3 * df_metrics['cal_har'] \
                       + 1/3 * df_metrics['dav_bould'] \
                       + 1/3 * df_metrics['silhouette']

        return union_metric.idxmax()
    
    
def plot_elbow(X, model, k=(2, 20)):
    """
    Метод локтя из библиотеки yellowbrick
    X: данные для обучения
    model: модель кластеризации
    k: диапазон кластеров для анализа
    """
    # обучаем кластеризатор
    k_min, k_max= k
    inertia = []
    for n in tqdm(range(k_min, k_max+1)):
        model.set_params(n_clusters=n)
        model.fit(X)
        inertia.append(model.inertia_)
        
    # определяем оптимальное число кластеров
    knee = KneeLocator(range(len(inertia)), inertia, 
                       curve_direction='decreasing',  
                       curve_nature='convex').elbow
    # рисуем
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(k_min, k_max+1), inertia, 'bo-')
    ax.set_title(f'Оптимальное число кластеров: {knee}', fontsize=16)
    ax.set_xlabel('Число кластеров', fontsize=14)
    ax.set_ylabel('inertia', fontsize=14)
    ax.set_xticks(range(k_min, k_max+1, 2))
    ax.tick_params(axis='both', labelsize=12)
    if knee:
        ax.axvline(knee, color='r', linestyle='--')
    plt.show()    

    
def plot_wordcloud(data: Union[np.array, list], labels: np.array, wc: WordCloud):
    """
    Генерация облака слов
    data: массив из строк
    labels: метки кластеров
    wc: экземляр класса WordCloud для генерации облака
    """
    assert len(data) == len(labels), 'Разная длина меток и данных'

    if isinstance(data, list):
        data = np.array(data)

    # фильтруем кластер помеченный как шум
    clusters = np.unique(labels)
    clusters = clusters[np.where(clusters != -1)[0]]

    # рисуем облако слов
    fig, axes = plt.subplots(len(clusters), 1, figsize=(15, 6 * len(clusters)))
    for cluster, ax in zip(clusters, axes):
        try:
            sample = data[labels == cluster]
            cloud = wc.generate(' '.join(sample))
            ax.imshow(cloud, interpolation='bilinear')
            ax.axis("off")
            ax.set_title(f'Кластер: {cluster}', fontsize=20, fontweight='bold', pad=5)
        except Exception as e:
            continue
#     return fig    


def gen_colors(seq_colors):
    assert len(seq_colors) != 0, 'Пустая последовательность'
    j = 0
    while 1:
        yield seq_colors[j]
        j += 1
        if j >= len(seq_colors):
            j = 0
            
def get_color_map(clusters, colors):
    gen = gen_colors(colors)
    color_map = {str(k): v for k, v, in zip(clusters, [next(gen) for _ in range(len(clusters))])}
    return color_map