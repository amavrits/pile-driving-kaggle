import numpy as np
import pandas as pd
import seaborn as sns

def add_location_distind(df):
    dist_dict = {'<500m': 0,
                 '500m - 1500m': 1,
                 '1500m - 3000m': 2,
                 '3000m - 4500m': 3,
                 '>4500m': 4}
    df['dist_indicator'] = [dist_dict[i] for i in df['Interdistance class']]
    # return df.pivot_table(index=['ID location 1'], columns='ID location 2', values='dist_indicator')
    df = df[df['dist_indicator'] != 0]
    return df

def find_idx_depth(z_finder, z_where):
    z_diff = np.abs(z_finder[:, None] - z_where.T)
    idx = np.argmin(z_diff, axis=1)
    return idx

def set_geometry_dict(df):
    return {pile: df[df['Location ID']==pile]['Diameter [m]'].values[0]
                 for pile in list(pd.unique(df['Location ID']))}

def amend_cpt_data(df_pile, df_cpt, ammended_columns=['qnet', 'friction']):
    dict_geom = set_geometry_dict(df_pile)
    df_pile[ammended_columns] = 0
    for (pile, geom) in dict_geom.items():
        z = df_cpt[df_cpt['ID'] == pile]['z [m]'].values
        fs = df_cpt[df_cpt['ID'] == pile]['fs [MPa]'].values
        qnet = df_cpt[df_cpt['ID'] == pile]['qnet [MPa]'].values
        friction = np.cumsum(fs[:-1] * np.pi * geom * np.diff(z))
        friction = np.append(friction, friction[-1])
        idx = find_idx_depth(z_finder=df_pile[df_pile['Location ID'] == pile]['z [m]'].values,
                             z_where=df_cpt[df_cpt['ID'] == pile]['z [m]'].values)
        df_pile.loc[df_pile['Location ID'] == pile, ammended_columns] = np.c_[qnet[idx], friction[idx]]
    return df_pile

def load_data(cpt_file, pile_file):
    df_cpt = pd.read_csv(cpt_file)
    df_pile = pd.read_csv(pile_file)
    df_cpt.dropna(how='any', inplace=True)
    df_cpt.reset_index(drop=True, inplace=True)
    df_pile.dropna(how='any', inplace=True)
    df_pile.reset_index(drop=True, inplace=True)
    df_pile = amend_cpt_data(df_pile=df_pile, df_cpt=df_cpt)
    df_pile.drop(columns=['ID'], inplace=True)
    rename_dict = {'Blowcount [Blows/m]': 'blowcount',
                   'z [m]': 'z',
                   'Diameter [m]': 'diameter',
                   'Number of blows': 'blownumber',
                   'Normalised ENTRHU [-]': 'entrhu',
                   'Normalised hammer energy [-]': 'hammer_energy',
                   'Location ID': 'ID'}
    df_pile.rename(rename_dict, inplace=True, axis='columns')
    return df_pile


if __name__ == '__main__':

    df_cpt = pd.read_csv('..\\data\\full_cpt_training_data_withnormalised.csv')
    df_pile = pd.read_csv('..\\data\\training_data.csv')
    df_geom = pd.read_csv('..\\data\\interdistance_data.csv')

    df_cpt.dropna(how='any', inplace=True)
    df_cpt.reset_index(drop=True, inplace=True)
    df_pile.dropna(how='any', inplace=True)
    df_pile.reset_index(drop=True, inplace=True)
    df_geom.dropna(how='any', inplace=True)
    df_geom.reset_index(drop=True, inplace=True)

    # df_geom = add_location_distind(df_geom)

    # sns.pairplot(data=df_pile[df_pile['Location ID']=='EK'], vars=['qc [MPa]', 'Normalised ENTRHU [-]', 'Blowcount [Blows/m]'])

    df_pile = amend_cpt_data(df_pile, df_cpt, ammended_columns=['qnet', 'friction'])




