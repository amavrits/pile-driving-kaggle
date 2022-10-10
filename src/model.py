import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from data_import import load_data, find_idx_depth


if __name__ == '__main__':

    cpt_file = '..\\data\\full_cpt_training_data_withnormalised.csv'
    pile_file = '..\\data\\training_data.csv'

    df = load_data(cpt_file=cpt_file, pile_file=pile_file, window_length=None, conv_type='average')


    df = df[df['ID'] == 'EK']


    # sns.pairplot(data=df, vars=['qnet', 'friction', 'Ic', 'base_res_plug', 'base_res_wall', 'entrhu', 'blowcount'])


    # from sklearn.linear_model import LinearRegression
    # from sklearn.metrics import mean_squared_error
    #
    # X = df[['qnet', 'friction', 'entrhu']].values
    # y = df['blowcount'].values.reshape(-1, 1)
    # reg = LinearRegression().fit(X, y)
    # y_pred = reg.predict(X)
    # reg.score(X, y)
    # A = mean_squared_error(y, y_pred, squared=False)


    from EM_algorithm import EM
    from sklearn.cluster import KMeans
    from scipy.stats import norm
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error

    class EM_GMR(EM):

        def log_like_fn(self, par):
            x = self.X_train[:, :-1]
            y = self.X_train[:, -1]
            y_hat = x.dot(par[0])
            log_like = norm.logpdf(y, loc=y_hat, scale=par[1])
            return log_like


    def init_fn(X_train, n_clusters, Z=None):
        x = X_train[:, :-1].reshape(-1, 1)
        y = X_train[:, -1]
        if Z is None:
            kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(np.c_[x, y])
            Z = kmeans.labels_
        betas = np.zeros((n_clusters, x.shape[1] + 1))
        sigma = np.zeros(n_clusters)
        for i_cluster in range(n_clusters):
            idx = np.where(Z == i_cluster)[0]
            reg = LinearRegression().fit(x[idx], y[idx])
            y_hat = reg.predict(x[idx])
            residuals = (y[idx] - y_hat) ** 2
            sigma[i_cluster] = np.sqrt(np.dot(residuals.T, residuals) / (x.shape[0] - x.shape[1]))
            betas[i_cluster] = np.array([reg.intercept_, reg.coef_[0]])
        mix_par_init = (betas, sigma)
        mix_weights_init = np.array([sum(Z == i_cluster) for i_cluster in range(n_clusters)]) / x.shape[0]
        return mix_par_init, mix_weights_init

    data = np.c_[np.ones(len(df)), df[['friction', 'blowcount']].values]
    em = EM_GMR(init_fn=init_fn, model_type='linear')
    # em.EM_set_initialization(data, n_clusters=3, init_method='random')
    # em.train()
    em.multi_random_init(data, n_clusters=3, n_random_inits=100)


