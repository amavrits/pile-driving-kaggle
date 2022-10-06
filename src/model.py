import numpy as np
import pandas as pd
import seaborn as sns
from data_import import load_data, find_idx_depth


if __name__ == '__main__':

    cpt_file = '..\\data\\full_cpt_training_data_withnormalised.csv'
    pile_file = '..\\data\\training_data.csv'

    df_pile = load_data(cpt_file=cpt_file, pile_file=pile_file)


    df_pile['qnet_sum'] = 0
    # for pile in list(pd.unique(df_pile['ID'])):
    for pile in ['EK']:
        dummy = df_pile[df_pile['ID'] == pile]
        qnet = dummy['qnet'].values
        z = dummy['z'].values
        d = dummy['diameter'].values[0]
        qnet_sum = np.zeros_like(qnet)
        for i, zz in enumerate(z):
            idx = np.where(z<= zz+d)[0]
            qnet_sum[i] = np.sum(qnet[idx])
        df_pile.loc[df_pile['ID'] == pile, 'qnet_sum'] = qnet_sum


    df_pile = df_pile[df_pile['qnet_sum']!=0]


    sns.pairplot(data=df_pile, vars=['qnet', 'qnet_sum', 'friction', 'blowcount', 'entrhu'])


    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error

    X = df_pile[['qnet_sum', 'friction']].values
    y = df_pile['blowcount'].values.reshape(-1, 1)
    reg = LinearRegression().fit(X, y)
    y_pred = reg.predict(X)
    reg.score(X, y)
    A = mean_squared_error(y, y_pred, squared=False)

