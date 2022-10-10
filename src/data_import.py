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
    return {pile: (
        df[df['ID'] == pile]['diameter'].values[0],
        df[df['ID'] == pile]['Bottom wall thickness [mm]'].values[0]
    ) for pile in list(pd.unique(df['ID']))}

def amend_cpt_data(df_pile, df_cpt, ammended_columns=['qnet', 'friction', 'base_res_plug', 'base_res_wall', 'Ic'],
                   window_length=None, conv_type='custom'):
    dict_geom = set_geometry_dict(df_pile)
    df_pile[ammended_columns] = 0
    for (pile, geom) in dict_geom.items():
        z = df_cpt[df_cpt['ID'] == pile]['z [m]'].values
        fs = df_cpt[df_cpt['ID'] == pile]['fs [MPa]'].values
        qnet = df_cpt[df_cpt['ID'] == pile]['qnet [MPa]'].values
        area_base_plug = np.pi * (geom[0] / 2) ** 2
        area_base_wall = np.pi * (geom[0] / 2) ** 2 - np.pi * ((geom[0] - geom[1] / 1_000) / 2) ** 2
        friction = np.cumsum(fs[:-1] * np.pi * geom[0] * np.diff(z))
        friction = np.pad(friction, (0, 1), 'constant', constant_values=(0, friction[-1]))
        base_resistance_plug = qnet * area_base_plug
        base_resistance_wall = qnet * area_base_wall

        if window_length is None:
            window_length = int(geom[0] / np.diff(df_cpt['z [m]'].values)[0])
        qnet = convolve_input(x=qnet, window_length=window_length, conv_type=conv_type)
        base_resistance_plug = convolve_input(x=base_resistance_plug, window_length=window_length, conv_type=conv_type)
        base_resistance_wall = convolve_input(x=base_resistance_wall, window_length=window_length, conv_type=conv_type)

        idx = find_idx_depth(z_finder=df_pile[df_pile['ID'] == pile]['z'].values,
                             z_where=df_cpt[df_cpt['ID'] == pile]['z [m]'].values)

        df_pile.loc[df_pile['ID'] == pile, ammended_columns] = np.c_[qnet[idx],
                                                                     friction[idx],
                                                                     base_resistance_plug[idx],
                                                                     base_resistance_wall[idx],
                                                                     df_cpt['Ic [-]'].values[idx]]

    return df_pile

def convolve_pad(x, w):
    y = np.convolve(x, w, mode='valid')
    y = np.pad(y, (0, x.size-y.size), 'constant', constant_values=(0, y[-1]))
    return y

def convolve_input(x, w=None, window_length=None, conv_type='custom'):
    if conv_type == 'sum':
        w = np.ones(window_length)
        y = convolve_pad(x, w)
    elif conv_type == 'average':
        w = np.ones(window_length) / window_length
        y = convolve_pad(x, w)
    elif conv_type == 'max':
        dummy = np.tile(x.reshape(-1, 1), reps=(1, window_length))
        for i in range(window_length):
            dummy[:, i] = np.roll(dummy[:, i], -i)
        y = dummy.max(axis=1)
    else:
        if w is None:
            raise Exception('No custom weight given for convolution')
        y = convolve_pad(x, w)
        pass
    return y

def load_data(cpt_file, pile_file, window_length=None, conv_type='custom'):
    df_cpt = pd.read_csv(cpt_file)
    df_pile = pd.read_csv(pile_file)
    df_cpt.dropna(how='any', inplace=True)
    df_cpt.reset_index(drop=True, inplace=True)
    df_pile.dropna(how='any', inplace=True)
    df_pile.reset_index(drop=True, inplace=True)

    df_pile.drop(columns=['ID'], inplace=True)
    rename_dict = {'Blowcount [Blows/m]': 'blowcount',
                   'z [m]': 'z',
                   'Diameter [m]': 'diameter',
                   'Number of blows': 'blownumber',
                   'Normalised ENTRHU [-]': 'entrhu',
                   'Normalised hammer energy [-]': 'hammer_energy',
                   'Location ID': 'ID'}
    df_pile.rename(rename_dict, inplace=True, axis='columns')

    df_pile = amend_cpt_data(df_pile=df_pile, df_cpt=df_cpt, window_length=window_length, conv_type=conv_type)

    return df_pile


if __name__ == '__main__':

    cpt_file = '..\\data\\full_cpt_training_data_withnormalised.csv'
    pile_file = '..\\data\\training_data.csv'

    df_pile = load_data(cpt_file=cpt_file, pile_file=pile_file, conv_type='sum')



