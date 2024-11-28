import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# 1
df = pd.read_csv('medical_examination.csv')

# 2
df['overweight'] = ((df['weight'] / ((df['height'] / 100) ** 2)) > 25).astype('int8')

# 3
df[['cholesterol', 'gluc']] = (df[['cholesterol', 'gluc']] > 1).astype(int)

# 4
def draw_cat_plot():

    # 5
    df_cat = pd.melt(df,
                     id_vars=['cardio'],
                     value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'],
                     var_name='variable',
                     value_name='value')

    # 6
    df_cat_grouped = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')

    # 7
    x = sns.catplot(data=df_cat_grouped, 
                    x="variable",
                    y="total",
                    hue="value", 
                    col="cardio", 
                    kind="bar" 
                    )  

    x.set_axis_labels("variable", "total")
    x.set_titles("cardio = {col_name}")

    # 8
    fig = x.fig

    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    df_heat = df[
                 (df['ap_lo'] <= df['ap_hi'])                   &
                 (df['height'] >= df['height'].quantile(0.025)) &
                 (df['height'] <= df['height'].quantile(0.975)) &
                 (df['weight'] >= df['weight'].quantile(0.025)) &
                 (df['weight'] <= df['weight'].quantile(0.975)) 
                ]

    # 12
    corr = df_heat.corr(method='pearson', min_periods=1, numeric_only=True)

    # 13
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14
    fig, ax = plt.subplots(figsize=(10,8))

    # 15
    #colors = ["blue", "black", "darkblue", "darkred", "red", "orange"]  # Verlauf von schwarz, blau, rot, orange
    #n_bins = 100  # Anzahl der Farbverläufe
    #cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=n_bins)
    sns.heatmap(corr, mask=mask, annot=True, fmt='.1f', cbar_kws={'shrink': 0.8}, ax=ax)

    # 16
    fig.savefig('heatmap.png')
    return fig
