#Imports
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
sns.set() # Setting seaborn as default style even if use only matplotlib
plt.rcParams.update({"font.family": "Roboto Condensed"})
import graphs

#Load data
df_rasp = pd.read_pickle("rasp_result_time.pkl")
df_pi_energy_mean = pd.read_pickle("rasp_energy_consumption.pkl")


#%%
#Mean
df_transmission = df_rasp.loc[(df_rasp['iteration'] == '50'), ['n_devices', 'n_sensors', 'trans_to_device_time']]
df_transmission['n_devices'] = df_transmission['n_devices'].astype(int)
df_transmission_mean = df_transmission.groupby(['n_devices', 'n_sensors'])[['trans_to_device_time']].mean().reset_index().copy()
df_transmission_mean.set_index(['n_devices', 'n_sensors'], inplace=True)

df_rasp.drop(df_transmission.index, inplace=True)
df_rasp.drop(columns=['trans_to_device_time'], inplace=True)
df_rasp['n_devices'] = df_rasp['n_devices'].astype(int)

df_rasp_mean_by_sensor = df_rasp.groupby(['n_devices', 'n_sensors'])[['local_training_time', 'rec_from_manager_time', 'trans_to_manager_time']].mean().reset_index().copy()
df_rasp_mean_by_sensor.set_index(['n_devices', 'n_sensors'], inplace=True)
df_rasp_mean_by_sensor = df_rasp_mean_by_sensor.join(df_transmission_mean)

global_model_update_interval = [0.125, 0.25, 0.5, 1, 2, 4, 8, 24] #horas
n_devices = list(df_rasp_mean_by_sensor.index.unique(level=0))
n_rounds = 50

energy_daily_Raspi = pd.DataFrame()

MAX_DAILY_TIME = 86400
V = 5 #5V for Raspberry Pi

for index, update_interval in enumerate(global_model_update_interval):
    training_time = (24/update_interval) * n_rounds * df_rasp_mean_by_sensor['local_training_time']
    rec_global_model_time = (24/update_interval) * n_rounds * df_rasp_mean_by_sensor['rec_from_manager_time']
    trans_model_to_manager_time = (24/update_interval) * n_rounds * df_rasp_mean_by_sensor['trans_to_manager_time']
    trans_model_to_devices_time = (24/update_interval) * n_rounds * df_rasp_mean_by_sensor['trans_to_device_time']
    idle_time = MAX_DAILY_TIME - (training_time + rec_global_model_time + trans_model_to_manager_time + trans_model_to_devices_time)
    idle_time.rename("idle_time", inplace=True)

    idle_energy = idle_time * df_pi_energy_mean.loc[(df_pi_energy_mean['Type'] == 'idle'), 'Current'].item() * V
    training_energy = training_time * df_pi_energy_mean.loc[(df_pi_energy_mean['Type'] == 'local_training_time'), 'Current'].item() * V
    rec_global_model_energy = rec_global_model_time * df_pi_energy_mean.loc[(df_pi_energy_mean['Type'] == 'rec_from_manager_time'), 'Current'].item() * V
    trans_model_to_manager_energy = trans_model_to_manager_time * df_pi_energy_mean.loc[(df_pi_energy_mean['Type'] == 'trans_to_manager_time'), 'Current'].item() * V
    trans_model_to_devices_energy = trans_model_to_devices_time * df_pi_energy_mean.loc[(df_pi_energy_mean['Type'] == 'trans_to_device_time'), 'Current'].item() * V

    temp = pd.concat([idle_energy, training_energy, rec_global_model_energy, trans_model_to_manager_energy, trans_model_to_devices_energy], axis=1)
    temp['global_model_update_interval'] = update_interval
    energy_daily_Raspi = pd.concat([energy_daily_Raspi,temp])

energy_daily_Raspi.rename(columns={"idle_time":"idle_mJ",
                           "local_training_time":"local_training_mJ",
                           "rec_from_manager_time":"rec_from_manager_mJ",
                           "trans_to_manager_time":"trans_to_manager_mJ",
                           "trans_to_device_time":"trans_to_device_mJ"}, inplace=True)
print(f'Daily energy rasp: \n {energy_daily_Raspi}')


graphs.plot_rasp_energy_total(energy_daily_Raspi)

##########################################

#########
# Percentual inativo comparando 7,5 e 15 minutos com o restante. Comparativo do total de energia gasta
percent = energy_daily_Raspi.reset_index().copy()
percent = percent.set_index(['global_model_update_interval', 'n_sensors', 'n_devices'])
percent = percent.groupby('global_model_update_interval').mean().sum(axis=1).reset_index()
print("percentual de aumento para 7,5 minutos (0.125): ", (1-(percent[0][7] / percent[0][0])) * 100)
print("percentual de aumento para 15 minutos (0.25): ", (1-(percent[0][7] / percent[0][1])) * 100)



#########
# Percentual comparativo do tempo de inatividade (idle) com o restante do tempo ativo, com
# global_model_update_interval igual ou superior a 0,5 (30 minutos)
percent = energy_daily_Raspi.reset_index().copy()
percent.drop( percent.loc[((percent['global_model_update_interval'] == 0.125) | (percent['global_model_update_interval'] == 0.25))].index, inplace=True )
percent.drop(columns=['n_devices', 'n_sensors', 'global_model_update_interval'], inplace=True)
percent['active_mJ'] = percent['local_training_mJ'] + percent['rec_from_manager_mJ'] + percent['trans_to_manager_mJ'] + percent['trans_to_device_mJ']
percent.drop(columns=['local_training_mJ', 'rec_from_manager_mJ', 'trans_to_manager_mJ', 'trans_to_device_mJ'], inplace=True)
print("Percentual em idle: ", (1 - (percent['active_mJ'].mean() / percent['idle_mJ'].mean())) * 100)

#########
percent = energy_daily_Raspi.reset_index().copy()
percent = percent.set_index(['global_model_update_interval', 'n_sensors', 'n_devices'])
# percent = percent.sum(axis=1).reset_index().set_index('global_model_update_interval')
percent = percent.groupby('global_model_update_interval').mean().sum(axis=1).reset_index()
print("percentual de aumento para 7,5 minutos (0.125): ", (1-(percent[0][7] / percent[0][0])) * 100)
print("percentual de aumento para 15 minutos (0.25): ", (1-(percent[0][7] / percent[0][1])) * 100)
