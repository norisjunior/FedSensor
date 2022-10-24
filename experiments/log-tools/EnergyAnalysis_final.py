#Imports
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
sns.set() # Setting seaborn as default style even if use only matplotlib
plt.rcParams.update({"font.family": "Roboto Condensed"})
import graphs

#Load data
energy_32001 = pd.read_pickle("energy_32001.pkl")
energy_MLModel = pd.read_pickle("energy_MLModel.pkl")
energy_predict = pd.read_pickle("energy_predict.pkl")
energy_idle = pd.read_pickle("energy_lifetime.pkl")



#%%
#Energy 32001 plots


#!!!!!!! SEÇÃO DA TESE: CAPÍTULO 6, SEÇÃO 1: Consumo de energia para estruturação de sensores e variáveis dos modelos nos dispositivos IoT ultra-restritos

energy_32001_mean = energy_32001.groupby(['Device', 'Dataset', 'SensorsNumber', 'Message'])[['CPUmj', 'LPMmj', 'DEEPLPMmj', 'TXmj', 'RXmj']].mean().reset_index().copy()
#energy_32001_mean.loc[:,'TXmj'] = 0.001
energy_32001_mean.loc[energy_32001_mean['Device'] == 'CC1352P1','TXmj'] = 0.001


list_sum = ['CPUmj', 'LPMmj', 'DEEPLPMmj', 'TXmj', 'RXmj']
energy_32001_mean.loc[:, 'FinalmJ'] = energy_32001_mean.loc[:, list_sum].sum(axis=1)

for i in energy_32001_mean.index:
    graphs.plot_circular_barplot(
        device=energy_32001_mean.loc[( (energy_32001_mean.index == i) ), ['Device']].values.item(),
        sensorsNumber=energy_32001_mean.loc[( (energy_32001_mean.index == i) ), ['SensorsNumber']].values.item(),
        labels= ['RX', 'CPU', 'LPM', 'TX'],
        data=energy_32001_mean.loc[( (energy_32001_mean.index == i) ), ['RXmj', 'CPUmj', 'LPMmj', 'TXmj']].values[0].tolist())

################
# CPU percentage related to RX
CPU_perc = energy_32001_mean['CPUmj']/energy_32001_mean['RXmj']
CPU_perc.mean()
CPU_perc.std()


#######################
#3, 4 and 9 sensors increased percentage compared to 2 sensors by device
percent_comparison = energy_32001_mean.set_index('Device')



for device in list(percent_comparison.index.unique()):
    for index, sensorNumber in enumerate(list(percent_comparison.loc[device, 'SensorsNumber'])):
        two = percent_comparison.loc[( (percent_comparison.index == device) & (percent_comparison['SensorsNumber'] == '2') ), ['FinalmJ']].values.item()
        other = percent_comparison.loc[( (percent_comparison.index == device) & (percent_comparison['SensorsNumber'] == sensorNumber) ), ['FinalmJ']].values.item()
        print('device ' + device + ' - sensorNumber ' + sensorNumber + ': ' + str(round((((other/two) - 1) * 100),2)) + ' %')
















#%%
#Energy MLModel plots

#!!!!!!! SEÇÃO DA TESE: CAPÍTULO 6, SEÇÃO 2: Consumo de energia para recebimento do modelo de ML global dispositivos IoT ultra-restritos


energy_MLModel_mean = energy_MLModel.groupby(['Device', 'Dataset', 'MLModel', 'ResultName', 'ResultNumber', 'SensorsNumber'])[['CPUmj', 'LPMmj', 'DEEPLPMmj', 'TXmj', 'RXmj']].mean().reset_index().copy()
energy_MLModel_mean.loc[:,'TXmj'] = 0

list_sum = ['CPUmj', 'LPMmj', 'DEEPLPMmj', 'TXmj', 'RXmj']
energy_MLModel_mean.loc[:, 'FinalmJ'] = energy_MLModel_mean.loc[:, list_sum].sum(axis=1)

energy_MLModel_mean.loc[energy_MLModel_mean['MLModel'] == "32103", 'ModelName'] = "Regressão logística"
energy_MLModel_mean.loc[energy_MLModel_mean['MLModel'] == "32106", 'ModelName'] = "k-means"
energy_MLModel_mean.loc[energy_MLModel_mean['MLModel'] == "32102", 'ModelName'] = "Regressão linear"

energy_MLModel_mean.loc[energy_MLModel_mean['MLModel'] == "32103", 'ModelNameResultType'] = "classes"
energy_MLModel_mean.loc[energy_MLModel_mean['MLModel'] == "32106", 'ModelNameResultType'] = "grupos"

#############################################################################################################
#Gráficos do dataset MOTOR
energy_MLModel_MOTOR = energy_MLModel_mean.loc[( (energy_MLModel_mean['Dataset'] == 'MOTOR'))]

for resultName in energy_MLModel_MOTOR['ResultName'].unique():
    dataset = energy_MLModel_MOTOR[energy_MLModel_MOTOR['ResultName'] == resultName].copy()
    for resultMLModel in dataset['MLModel'].unique():
        graph_dataset = dataset[dataset['MLModel'] == resultMLModel].copy()
        graphs.plot_stacked_barplot_MLModel(graph_dataset)


#Mean percentage
percentage = energy_MLModel_MOTOR.loc[energy_MLModel_MOTOR['ModelName'] == 'k-means'].set_index('ResultNumber')
print('k-means: ' + str(round(((percentage.loc['3']['FinalmJ'].mean()/percentage.loc['2']['FinalmJ'].mean()) - 1)*100,2)) + ' %')

percentage = energy_MLModel_MOTOR.loc[energy_MLModel_MOTOR['ModelName'] == 'Regressão logística'].set_index('ResultNumber')
print('logreg: ' + str(round(((percentage.loc['3']['FinalmJ'].mean()/percentage.loc['2']['FinalmJ'].mean()) - 1)*100,2)) + ' %')


#Mean percentage by device
percentage = energy_MLModel_MOTOR.loc[energy_MLModel_MOTOR['ModelName'] == 'k-means'].set_index('Device')
print('k-means 2 sensores, 3 classes (FinalmJ): \n', round((((percentage.loc[percentage['ResultNumber'] == '3']['FinalmJ'] / percentage.loc[percentage['ResultNumber'] == '2']['FinalmJ']) - 1) * 100),2))


percentage = energy_MLModel_MOTOR.loc[energy_MLModel_MOTOR['ModelName'] == 'Regressão logística'].set_index('Device')
print('regressão logística 2 sensores, 3 classes (FinalmJ): \n', round((((percentage.loc[percentage['ResultNumber'] == '3']['FinalmJ'] / percentage.loc[percentage['ResultNumber'] == '2']['FinalmJ']) - 1) * 100),2))
print('regressão logística 2 sensores, 3 classes (RXmj): \n', round((((percentage.loc[percentage['ResultNumber'] == '3']['RXmj'] / percentage.loc[percentage['ResultNumber'] == '2']['RXmj']) - 1) * 100),2))
print('regressão logística 2 sensores, 3 classes (CPUmj): \n', round((((percentage.loc[percentage['ResultNumber'] == '3']['CPUmj'] / percentage.loc[percentage['ResultNumber'] == '2']['CPUmj']) - 1) * 100),2))




#############################################################################################################




#############################################################################################################
#Gráficos do dataset AQI

#Regressão Linear
energy_MLModel_AQI_linreg = energy_MLModel_mean.loc[( (energy_MLModel_mean['Dataset'] == 'AQI') & (energy_MLModel_mean['MLModel'] == '32102') )]

for resultSensors in energy_MLModel_AQI_linreg['SensorsNumber'].unique():
    graph_dataset = energy_MLModel_AQI_linreg[energy_MLModel_AQI_linreg['SensorsNumber'] == resultSensors].copy()
    graphs.plot_stacked_barplot_MLModel(graph_dataset)

percentage = energy_MLModel_AQI_linreg.set_index('SensorsNumber')
print('4: ' + str(round(((percentage.loc['4']['FinalmJ'].mean()/percentage.loc['2']['FinalmJ'].mean()) - 1)*100,2)) + ' %')
print('9: ' + str(round(((percentage.loc['9']['FinalmJ'].mean()/percentage.loc['2']['FinalmJ'].mean()) - 1)*100,2)) + ' %')







#####################

#Regressão logística e k-means
energy_MLModel_AQI_others = energy_MLModel_mean.loc[( (energy_MLModel_mean['Dataset'] == 'AQI') & (energy_MLModel_mean['MLModel'] != '32102') )]

for resultName in energy_MLModel_AQI_others['ResultName'].unique():
    dataset = energy_MLModel_AQI_others[energy_MLModel_AQI_others['ResultName'] == resultName].copy()
    for resultMLModel in dataset['MLModel'].unique():
        filtered_dataset = dataset[dataset['MLModel'] == resultMLModel].copy()
        for resultSensors in filtered_dataset['SensorsNumber'].unique():
            mlmodel_dataset = filtered_dataset[filtered_dataset['SensorsNumber'] == resultSensors].copy()

            if (resultName == 'binary'):
                graph_dataset = mlmodel_dataset.copy()
                graphs.plot_stacked_barplot_MLModel(graph_dataset)
            else:
                for resultNumber in mlmodel_dataset['ResultNumber'].unique():
                    graph_dataset = mlmodel_dataset[mlmodel_dataset['ResultNumber'] == resultNumber].copy()
                    graphs.plot_stacked_barplot_MLModel(graph_dataset)

#!!!!!!!#####Regressão logística
#!!!!!!!#####2 sensores!
#Mean by device
percentage = energy_MLModel_AQI_others.loc[ (energy_MLModel_AQI_others['ModelName'] == 'Regressão logística') & (energy_MLModel_AQI_others['SensorsNumber'] == '2') ].set_index('Device')
print('regressão logística 2 sensores, 3 classes (FinalmJ): \n', round((((percentage.loc[percentage['ResultNumber'] == '3']['FinalmJ'] / percentage.loc[percentage['ResultNumber'] == '2']['FinalmJ']) - 1) * 100),2))
print('regressão logística 2 sensores, 6 classes (FinalmJ): \n', round((((percentage.loc[percentage['ResultNumber'] == '6']['FinalmJ'] / percentage.loc[percentage['ResultNumber'] == '2']['FinalmJ']) - 1) * 100),2))

#Mean by device and resouce
print('regressão logística 2 sensores, 3 classes (RX): \n', round((((percentage.loc[percentage['ResultNumber'] == '3']['RXmj'] / percentage.loc[percentage['ResultNumber'] == '2']['RXmj']) - 1) * 100),2))
print('regressão logística 2 sensores, 6 classes (CPU): \n', round((((percentage.loc[percentage['ResultNumber'] == '6']['CPUmj'] / percentage.loc[percentage['ResultNumber'] == '2']['CPUmj']) - 1) * 100),2))


#Comparação entre diferentes números de sensores para  o mesmo número de classe (2 classes)
#Mean by device
percentage = energy_MLModel_AQI_others.loc[ (energy_MLModel_AQI_others['ModelName'] == 'Regressão logística') & (energy_MLModel_AQI_others['ResultNumber'] == '2') ].set_index('Device')
print('regressão logística 4 sensores, 2 classes (FinalmJ): \n', round((((percentage.loc[percentage['SensorsNumber'] == '4']['FinalmJ'] / percentage.loc[percentage['SensorsNumber'] == '2']['FinalmJ']) - 1) * 100),2))
print('regressão logística 9 sensores, 2 classes (FinalmJ): \n', round((((percentage.loc[percentage['SensorsNumber'] == '9']['FinalmJ'] / percentage.loc[percentage['SensorsNumber'] == '2']['FinalmJ']) - 1) * 100),2))

#Mean by device and resouce
print('regressão logística 4 sensores, 2 classes (RX): \n', round((((percentage.loc[percentage['SensorsNumber'] == '4']['RXmj'] / percentage.loc[percentage['SensorsNumber'] == '2']['RXmj']) - 1) * 100),2))
print('regressão logística 4 sensores, 2 classes (CPU): \n', round((((percentage.loc[percentage['SensorsNumber'] == '4']['CPUmj'] / percentage.loc[percentage['SensorsNumber'] == '2']['CPUmj']) - 1) * 100),2))

print('regressão logística 9 sensores, 2 classes (RX): \n', round((((percentage.loc[percentage['SensorsNumber'] == '9']['RXmj'] / percentage.loc[percentage['SensorsNumber'] == '2']['RXmj']) - 1) * 100),2))
print('regressão logística 9 sensores, 2 classes (CPU): \n', round((((percentage.loc[percentage['SensorsNumber'] == '9']['CPUmj'] / percentage.loc[percentage['SensorsNumber'] == '2']['CPUmj']) - 1) * 100),2))


#!!!!!!!#####k-means
#!!!!!!!#####2 sensores!
#Mean by device
percentage = energy_MLModel_AQI_others.loc[ (energy_MLModel_AQI_others['ModelName'] == 'k-means') & (energy_MLModel_AQI_others['SensorsNumber'] == '2') ].set_index('Device')
print('k-means 2 sensores, 3 classes (FinalmJ): \n', round((((percentage.loc[percentage['ResultNumber'] == '3']['FinalmJ'] / percentage.loc[percentage['ResultNumber'] == '2']['FinalmJ']) - 1) * 100),2))
print('k-means 2 sensores, 6 classes (FinalmJ): \n', round((((percentage.loc[percentage['ResultNumber'] == '6']['FinalmJ'] / percentage.loc[percentage['ResultNumber'] == '2']['FinalmJ']) - 1) * 100),2))

#Mean by device and resouce
print('k-means 2 sensores, 3 classes (RX): \n', round((((percentage.loc[percentage['ResultNumber'] == '3']['RXmj'] / percentage.loc[percentage['ResultNumber'] == '2']['RXmj']) - 1) * 100),2))
print('k-means 2 sensores, 6 classes (CPU): \n', round((((percentage.loc[percentage['ResultNumber'] == '6']['CPUmj'] / percentage.loc[percentage['ResultNumber'] == '2']['CPUmj']) - 1) * 100),2))


#Comparação entre diferentes números de sensores para  o mesmo número de classe (2 classes)
#Mean by device
percentage = energy_MLModel_AQI_others.loc[ (energy_MLModel_AQI_others['ModelName'] == 'k-means') & (energy_MLModel_AQI_others['ResultNumber'] == '2') ].set_index('Device')
print('k-means 4 sensores, 2 classes (FinalmJ): \n', round((((percentage.loc[percentage['SensorsNumber'] == '4']['FinalmJ'] / percentage.loc[percentage['SensorsNumber'] == '2']['FinalmJ']) - 1) * 100),2))
print('k-means 9 sensores, 2 classes (FinalmJ): \n', round((((percentage.loc[percentage['SensorsNumber'] == '9']['FinalmJ'] / percentage.loc[percentage['SensorsNumber'] == '2']['FinalmJ']) - 1) * 100),2))

#Mean by device and resouce
print('k-means 4 sensores, 2 classes (RX): \n', round((((percentage.loc[percentage['SensorsNumber'] == '4']['RXmj'] / percentage.loc[percentage['SensorsNumber'] == '2']['RXmj']) - 1) * 100),2))
print('k-means 4 sensores, 2 classes (CPU): \n', round((((percentage.loc[percentage['SensorsNumber'] == '4']['CPUmj'] / percentage.loc[percentage['SensorsNumber'] == '2']['CPUmj']) - 1) * 100),2))

print('k-means 9 sensores, 2 classes (RX): \n', round((((percentage.loc[percentage['SensorsNumber'] == '9']['RXmj'] / percentage.loc[percentage['SensorsNumber'] == '2']['RXmj']) - 1) * 100),2))
print('k-means 9 sensores, 2 classes (CPU): \n', round((((percentage.loc[percentage['SensorsNumber'] == '9']['CPUmj'] / percentage.loc[percentage['SensorsNumber'] == '2']['CPUmj']) - 1) * 100),2))








#%%
#Energy predict plots

#!!!!!!! SEÇÃO DA TESE: CAPÍTULO 6, SEÇÃO 3: Consumo de energia para a realização de inferência pelos dispositivos IoT ultra-restritos


energy_predict_mean = energy_predict.groupby(['Device', 'Dataset', 'MLModel', 'ResultName', 'ResultNumber', 'SensorsNumber'])[['CPUmj', 'LPMmj', 'DEEPLPMmj', 'TXmj', 'RXmj']].mean().reset_index().copy()
energy_predict_mean.loc[:,'TXmj'] = 0
energy_predict_mean.loc[:,'RXmj'] = 0

list_sum = ['CPUmj', 'LPMmj', 'DEEPLPMmj', 'TXmj', 'RXmj']
energy_predict_mean.loc[:, 'FinalmJ'] = energy_predict_mean.loc[:, list_sum].sum(axis=1)


energy_predict_mean.loc[energy_predict_mean['MLModel'] == "32103", 'ModelName'] = "Regressão logística"
energy_predict_mean.loc[energy_predict_mean['MLModel'] == "32106", 'ModelName'] = "k-means"
energy_predict_mean.loc[energy_predict_mean['MLModel'] == "32102", 'ModelName'] = "Regressão linear"

energy_predict_mean.loc[energy_predict_mean['MLModel'] == "32103", 'ModelNameResultType'] = "classes"
energy_predict_mean.loc[energy_predict_mean['MLModel'] == "32106", 'ModelNameResultType'] = "grupos"


energy_predict_linreg = energy_predict_mean.loc[( (energy_predict_mean['MLModel'] == '32102') )]
for sensorsNumber in np.sort(energy_predict_linreg['SensorsNumber'].unique()):
    graph_dataset = energy_predict_linreg[energy_predict_linreg['SensorsNumber'] == sensorsNumber].copy()
    graphs.plot_barplot_predict(graph_dataset)

#Regressão logística e k-means
energy_predict_others = energy_predict_mean.loc[( (energy_predict_mean['MLModel'] != '32102') )]

for mlModel in energy_predict_others['MLModel'].unique():
    mlmodel_dataset = energy_predict_others[energy_predict_others['MLModel'] == mlModel].copy()
    for sensorsNumber in np.sort(mlmodel_dataset['SensorsNumber'].unique()):
        graph_dataset = mlmodel_dataset[mlmodel_dataset['SensorsNumber'] == sensorsNumber].copy()
        graphs.plot_lines_predict(graph_dataset)



#!!!!!!!#####Regressão logística
#!!!!!!!#####2 sensores!
#Mean by device
for i in ['2', '3', '4', '9']:
    percentage = energy_predict_others.loc[ (energy_predict_others['ModelName'] == 'Regressão logística') & (energy_predict_others['SensorsNumber'] == i) ].set_index('Device')

    print('regressão logística ' + str(i) + ' sensores, 3 classes (FinalmJ): \n', (round(((1-(percentage.loc[percentage['ResultNumber'] == '2']['FinalmJ'] / percentage.loc[percentage['ResultNumber'] == '3']['FinalmJ'])) * 100),2)).mean())
    print('regressão logística ' + str(i) + ' sensores, 6 classes (FinalmJ): \n', (round(((1-(percentage.loc[percentage['ResultNumber'] == '2']['FinalmJ'] / percentage.loc[percentage['ResultNumber'] == '6']['FinalmJ'])) * 100),2)).mean())

#!!!!!!!#####Regressão logística
#!!!!!!!#####2 sensores!
#Mean by device
for i in ['2', '3', '4', '9']:
    percentage = energy_predict_others.loc[ (energy_predict_others['ModelName'] == 'k-means') & (energy_predict_others['SensorsNumber'] == i) ].set_index('Device')


    print('k-means ' + str(i) + ' sensores, 3 classes (FinalmJ): \n', (round(((1-(percentage.loc[percentage['ResultNumber'] == '2']['FinalmJ'] / percentage.loc[percentage['ResultNumber'] == '3']['FinalmJ'])) * 100),2)).mean())
    print('k-means ' + str(i) + ' sensores, 6 classes (FinalmJ): \n', (round(((1-(percentage.loc[percentage['ResultNumber'] == '2']['FinalmJ'] / percentage.loc[percentage['ResultNumber'] == '6']['FinalmJ'])) * 100),2)).mean())



energy_lifetime = energy_idle.round(0)

energy_lifetime['FinalmJ'] = 0.0
energy_lifetime['LPMmj'] = energy_lifetime['LPMmj'] + energy_lifetime['DEEPLPMmj']
energy_lifetime.loc[:, 'DEEPLPMmj'] = 0
list_exptot_sum = ['CPUmj', 'TXmj', 'RXmj', 'LPMmj']
energy_lifetime['FinalmJ'] = energy_lifetime[list_exptot_sum].sum(axis=1)


# Energia fornecida por 2 pilhas AA:
#(2.85Ah ∗ 1.5V ∗ 3600secs. ∗ 2) = 30780 Joules

# a energia FinalmJ está em milijoule, por isso divide por 1000

aa_bat = 30780

#Lifetime in days without FedSensor
energy_lifetime = energy_lifetime.sort_values(by=['Device'])
energy_lifetime["Idle"] = (aa_bat / (energy_lifetime["FinalmJ"] / 1000) + 1).round(0)
energy_lifetime["Idle"] = energy_lifetime["Idle"].astype("int")
energy_lifetime_FedSensor = energy_lifetime[['Device', 'FinalmJ']].copy()
energy_lifetime_FedSensor.rename(columns = {'FinalmJ':'IdlemJ'}, inplace = True)
energy_lifetime_FedSensor = energy_lifetime_FedSensor.set_index("Device")



#Pegar a média de cada um dos energy_ anteriores: 32001, MLModel e predict (média por dispositivo do FinalmJ)...

#Dataframe com média do consumo de energia quando troca de sensores: 32001
energy_32001_mean_Device = energy_32001_mean.groupby(['Device'])[['FinalmJ']].mean().reset_index().copy()

#Dataframe com média do consumo de energia quando troca o modelo de ML global: MLModel
energy_MLModel_mean_Device = energy_MLModel_mean.groupby(['Device'])[['FinalmJ']].mean().reset_index().copy()
energy_MLModel_mean_Device_and_SensorsNumber = energy_MLModel_mean.groupby(['Device', 'SensorsNumber'])[['FinalmJ']].mean().reset_index().copy()

#Dataframe com média do consumo de energia quando faz a previsão: predict
energy_predict_mean_Device = energy_predict_mean.groupby(['Device'])[['FinalmJ']].mean().reset_index().copy()
energy_predict_mean_Device_and_SensorsNumber = energy_predict_mean.groupby(['Device', 'SensorsNumber'])[['FinalmJ']].mean().reset_index().copy()







#!!!!!!! SEÇÃO DA TESE: CAPÍTULO 6, SEÇÃO 4: Comparativo entre a recepção do modelo global e a realização de inferências


#Gerar o gráfico com 10 segundos e com 30 segundos
inference_interval = 30 # segundos
global_model_update_interval = [0.125, 0.25, 0.5, 1, 2, 4, 8, 24] #horas
devices = ['CC1352P1', 'Remote', 'Sensortag']

model_vs_pred = pd.DataFrame({
    'Device': [],
    'Interval': [],
    'global_update': [],
    'predict': [],
    })

model_vs_pred

for index, interval in enumerate(reversed(global_model_update_interval)):
    for device in reversed(devices):
        df2 = pd.DataFrame([[device,interval]], columns=['Device','Interval'])
        model_vs_pred = pd.concat([df2, model_vs_pred])

model_vs_pred.set_index("Device", inplace=True)

for index, interval in enumerate(reversed(global_model_update_interval)):
    model_vs_pred.loc[(model_vs_pred['Interval'] == interval),['global_update']] = ((24/interval) * energy_MLModel_mean_Device.set_index("Device")).rename(columns={"FinalmJ":"global_update"})
    model_vs_pred.loc[(model_vs_pred['Interval'] == interval),['predict']] = (86400/inference_interval * energy_predict_mean_Device.set_index("Device")).rename(columns={"FinalmJ":"predict"})

model_vs_pred['FinalmJ'] = model_vs_pred['global_update'] + model_vs_pred['predict']

model_vs_pred['global_update_perc'] = (model_vs_pred['global_update'] / model_vs_pred['FinalmJ']) * 100
model_vs_pred['predict_perc'] = (model_vs_pred['predict'] / model_vs_pred['FinalmJ']) * 100


model_vs_pred['bat_lifetime'] = 0
for index, interval in enumerate(reversed(global_model_update_interval)):
    # interval = 1
    model_vs_pred.loc[(model_vs_pred['Interval'] == interval),['bat_lifetime']] = (
        (aa_bat / ((model_vs_pred.loc[(model_vs_pred['Interval'] == interval),['FinalmJ']] + energy_lifetime_FedSensor.rename(columns={"IdlemJ":"FinalmJ"})) / 1000))
        ).rename(columns={"FinalmJ":"bat_lifetime"}).astype("int")

model_vs_pred = model_vs_pred.reset_index().rename(columns={model_vs_pred.index.name:'Device'})

graphs.plot_model_vs_predict(model_vs_pred, inference_interval)




#%%

# Gráfico de comparação da vida útil considerando diferentes intervalos de tomada de decisão,
# com as linhas de base do dispositivo em idle (linhas fixas, uma para cada dispositivo)

#!!!!!!! SEÇÃO DA TESE: CAPÍTULO 6, SEÇÃO 4:


#!!! Fazer o gráfico com 10 e com 30 segundos, igual ao que foi feito no gráfico anterior
prediction_interval = 10 #10 e 30 segundos
global_model_update_interval = [0.125, 0.25, 0.5, 1, 2, 4, 8, 24] #horas

data = {
        'Device': ['CC1352P1', 'Remote', 'Sensortag'],
        }
energy_daily_FedSensor = pd.DataFrame.from_dict(data)
bat_lifetime_FedSensor = pd.DataFrame.from_dict(data)
energy_daily_FedSensor = energy_daily_FedSensor.set_index("Device")
bat_lifetime_FedSensor = bat_lifetime_FedSensor.set_index("Device")


for index, update_interval in enumerate(global_model_update_interval):
    energy_daily_FedSensor[str(global_model_update_interval[index])] = (
        ((24/update_interval) * energy_MLModel_mean_Device.set_index("Device")) +
        (86400/prediction_interval * energy_predict_mean_Device.set_index("Device"))
        )
bat_lifetime_FedSensor = (
        ((aa_bat / ((energy_daily_FedSensor.apply(lambda x: x + energy_lifetime_FedSensor['IdlemJ'])) / 1000))).astype("int")
        )

#plot!
print("Lifetime (days): \n\n", bat_lifetime_FedSensor)


data = {
        'Device': ['CC1352P1', 'Remote', 'Sensortag'],
        }
idle_lifetime = pd.DataFrame.from_dict(data)
idle_lifetime = idle_lifetime.set_index("Device")

#global_model_update_interval = 1 hora (fixo em 1h)
for index, update_interval in enumerate(global_model_update_interval):
    idle_lifetime[str(global_model_update_interval[index])] = energy_lifetime.set_index("Device").loc[:, ['Idle']].rename(columns={"Idle":"Baseline"})

#Plot!
graphs.plot_lines_lifetime_comparison_update(bat_lifetime_FedSensor, idle_lifetime, prediction_interval)


print('Percentual médio de diminuição da vida útil: \n',((1 - (bat_lifetime_FedSensor / idle_lifetime)) * 100).round(2).mean(axis=1))





######################################################################



#!!! Gerar gráficos com update global a cada hora e a cada 1/4 de hora (15 minutos, 0.25)

prediction_interval = [2, 10, 30, 60, 240]
global_model_update_interval = 1 #1 e 0.25 horas

data = {
        'Device': ['CC1352P1', 'Remote', 'Sensortag'],
        }
energy_daily_FedSensor = pd.DataFrame.from_dict(data)
bat_lifetime_FedSensor = pd.DataFrame.from_dict(data)
energy_daily_FedSensor = energy_daily_FedSensor.set_index("Device")
bat_lifetime_FedSensor = bat_lifetime_FedSensor.set_index("Device")


for index, inference_interval in enumerate(prediction_interval):
    energy_daily_FedSensor[str(prediction_interval[index])] = (
        ((24/global_model_update_interval) * energy_MLModel_mean_Device.set_index("Device")) +
        (86400/inference_interval * energy_predict_mean_Device.set_index("Device"))
        )
bat_lifetime_FedSensor = (
        ((aa_bat / ((energy_daily_FedSensor.apply(lambda x: x + energy_lifetime_FedSensor['IdlemJ'])) / 1000))).astype("int")
        )

#plot!
print("Lifetime (days): \n\n", bat_lifetime_FedSensor)


data = {
        'Device': ['CC1352P1', 'Remote', 'Sensortag'],
        }
idle_lifetime = pd.DataFrame.from_dict(data)
idle_lifetime = idle_lifetime.set_index("Device")

#global_model_update_interval = 1 hora (fixo em 1h)
for index, inference_interval in enumerate(prediction_interval):
    idle_lifetime[str(prediction_interval[index])] = energy_lifetime.set_index("Device").loc[:, ['Idle']].rename(columns={"Idle":"Baseline"})

#Plot!
graphs.plot_lines_lifetime_comparison_predict(bat_lifetime_FedSensor, idle_lifetime, global_model_update_interval)


print('Percentual médio de diminuição da vida útil: \n',((1 - (bat_lifetime_FedSensor / idle_lifetime)) * 100).round(2))




#####







#%%
# FEATURE SELECTION

#!!!!!!! SEÇÃO DA TESE: CAPÍTULO 6, SEÇÃO 5:

#Energy 32001 9 and 2
energy_32001_9and2 = energy_32001.groupby(['Device', 'Dataset', 'SensorsNumber', 'Message'])[['CPUmj', 'LPMmj', 'DEEPLPMmj', 'TXmj', 'RXmj']].mean().reset_index().copy()
energy_32001_9and2.drop(energy_32001_9and2.loc[(energy_32001_9and2['SensorsNumber'] != '9') & (energy_32001_9and2['SensorsNumber'] != '2')].index, inplace=True)
energy_32001_9and2.drop(columns=['Dataset', 'Message'], inplace=True)
energy_32001_9and2.loc[energy_32001_9and2['Device'] == 'CC1352P1','TXmj'] = 0.001
list_sum = ['CPUmj', 'LPMmj', 'DEEPLPMmj', 'TXmj', 'RXmj']
energy_32001_9and2.loc[:, 'FinalmJ'] = energy_32001_9and2.loc[:, list_sum].sum(axis=1)
energy_32001_9and2.set_index(['Device', 'SensorsNumber'], inplace=True)
energy_32001_9and2.drop(columns=['CPUmj', 'LPMmj', 'DEEPLPMmj', 'TXmj', 'RXmj'], inplace=True)
########################################################################################

#Energy MLModel 9 and 2
energy_MLModel_9and2 = energy_MLModel.copy()
energy_MLModel_9and2.drop(energy_MLModel_9and2.loc[( (energy_MLModel_9and2['ResultName'] == 'regression') )].index, inplace=True)
energy_MLModel_9and2.drop(energy_MLModel_9and2.loc[(energy_MLModel_9and2['SensorsNumber'] != '9') & (energy_MLModel_9and2['SensorsNumber'] != '2')].index, inplace=True)
energy_MLModel_9and2.drop(columns=['Dataset', 'MLModel', 'ResultName', 'ResultNumber'], inplace=True)
energy_MLModel_9and2 = energy_MLModel_9and2.groupby(['Device', 'SensorsNumber'])[['CPUmj', 'LPMmj', 'DEEPLPMmj', 'TXmj', 'RXmj']].mean().reset_index().copy()
energy_MLModel_9and2.loc[:,'TXmj'] = 0
list_sum = ['CPUmj', 'LPMmj', 'DEEPLPMmj', 'TXmj', 'RXmj']
energy_MLModel_9and2.loc[:, 'FinalmJ'] = energy_MLModel_9and2.loc[:, list_sum].sum(axis=1)
energy_MLModel_9and2.set_index(['Device', 'SensorsNumber'], inplace=True)
energy_MLModel_9and2.drop(columns=['CPUmj', 'LPMmj', 'DEEPLPMmj', 'TXmj', 'RXmj'], inplace=True)
########################################################################################


# Energy predict 9 and 2
energy_predict_9and2 = energy_predict.copy()
energy_predict_9and2.drop(energy_predict_9and2.loc[( (energy_predict_9and2['ResultName'] == 'regression') )].index, inplace=True)
energy_predict_9and2.drop(energy_predict_9and2.loc[(energy_predict_9and2['SensorsNumber'] != '9') & (energy_predict_9and2['SensorsNumber'] != '2')].index, inplace=True)
energy_predict_9and2.drop(columns=['Dataset', 'MLModel', 'ResultName', 'ResultNumber'], inplace=True)
energy_predict_9and2 = energy_predict_9and2.groupby(['Device', 'SensorsNumber'])[['CPUmj', 'LPMmj', 'DEEPLPMmj', 'TXmj', 'RXmj']].mean().reset_index().copy()
energy_predict_9and2.loc[:,'TXmj'] = 0
list_sum = ['CPUmj', 'LPMmj', 'DEEPLPMmj', 'TXmj', 'RXmj']
energy_predict_9and2.loc[:, 'FinalmJ'] = energy_predict_9and2.loc[:, list_sum].sum(axis=1)
energy_predict_9and2.set_index(['Device', 'SensorsNumber'], inplace=True)
energy_predict_9and2.drop(columns=['CPUmj', 'LPMmj', 'DEEPLPMmj', 'TXmj', 'RXmj'], inplace=True)
########################################################################################


# Energy idle
energy_idle_devices = energy_idle.round(0).copy()
energy_idle_devices['FinalmJ'] = 0.0
energy_idle_devices['LPMmj'] = energy_idle_devices['LPMmj'] + energy_idle_devices['DEEPLPMmj']
energy_idle_devices.loc[:, 'DEEPLPMmj'] = 0
list_exptot_sum = ['CPUmj', 'TXmj', 'RXmj', 'LPMmj']
energy_idle_devices['FinalmJ'] = energy_idle_devices[list_exptot_sum].sum(axis=1)
energy_idle_devices.drop(columns=['CPUmj', 'LPMmj', 'DEEPLPMmj', 'TXmj', 'RXmj'], inplace=True)
# Energia fornecida por 2 pilhas AA:
#(2.85Ah ∗ 1.5V ∗ 3600secs. ∗ 2) = 30780 Joules
# a energia FinalmJ está em milijoule, por isso divide por 1000
aa_bat = 30780
#Lifetime in days without FedSensor
energy_idle_devices = energy_idle_devices.sort_values(by=['Device'])
energy_idle_devices.set_index('Device', inplace=True)

prediction_interval = [2, 10, 30, 60, 240] #2 e 60 segundos
global_model_update_interval = 1 #horas

energy_9and2_total = pd.DataFrame()
idle_lifetime = pd.DataFrame()

for index, inference_interval in enumerate(prediction_interval):
    idle_lifetime[str(prediction_interval[index])] = ((aa_bat / (energy_idle_devices /1000) + 1).round(0)).astype(int)
    temp = energy_idle_devices + \
                    ( (24/global_model_update_interval) *  energy_MLModel_9and2 ) + \
                    ( (86400/inference_interval) *  energy_predict_9and2 )
    temp.rename(columns={"FinalmJ":str(inference_interval)}, inplace=True)
    energy_9and2_total = pd.concat([energy_9and2_total,temp], axis = 1)
    bat_graph_dataset = (aa_bat / (energy_9and2_total / 1000)).round(0)


graphs.plot_lines_lifetime_comparison_9and2(bat_graph_dataset, idle_lifetime)










#%%
# FEATURE SELECTION

# separando por classes

# Plotando 6 e 2 classes/grupos

#Energy 32001 6 and 2
energy_32001_6and2 = energy_32001.groupby(['Device'])[['CPUmj', 'LPMmj', 'DEEPLPMmj', 'TXmj', 'RXmj']].mean().reset_index().copy()
energy_32001_6and2.loc[energy_32001_6and2['Device'] == 'CC1352P1','TXmj'] = 0.001
list_sum = ['CPUmj', 'LPMmj', 'DEEPLPMmj', 'TXmj', 'RXmj']
energy_32001_6and2.loc[:, 'FinalmJ'] = energy_32001_6and2.loc[:, list_sum].sum(axis=1)
energy_32001_6and2.drop(columns=['CPUmj', 'LPMmj', 'DEEPLPMmj', 'TXmj', 'RXmj'], inplace=True)
energy_32001_6and2.set_index('Device', inplace=True)
########################################################################################

#Energy MLModel 6 and 2
energy_MLModel_6and2 = energy_MLModel.copy()
energy_MLModel_6and2.drop(energy_MLModel_6and2.loc[( (energy_MLModel_6and2['ResultName'] == 'regression') )].index, inplace=True)
energy_MLModel_6and2.drop(energy_MLModel_6and2.loc[(energy_MLModel_6and2['ResultNumber'] != '6') & (energy_MLModel_6and2['ResultNumber'] != '2')].index, inplace=True)
energy_MLModel_6and2.drop(columns=['Dataset', 'MLModel', 'ResultName', 'SensorsNumber'], inplace=True)
energy_MLModel_6and2 = energy_MLModel_6and2.groupby(['Device', 'ResultNumber'])[['CPUmj', 'LPMmj', 'DEEPLPMmj', 'TXmj', 'RXmj']].mean().reset_index().copy()
energy_MLModel_6and2.loc[:,'TXmj'] = 0
list_sum = ['CPUmj', 'LPMmj', 'DEEPLPMmj', 'TXmj', 'RXmj']
energy_MLModel_6and2.loc[:, 'FinalmJ'] = energy_MLModel_6and2.loc[:, list_sum].sum(axis=1)
energy_MLModel_6and2.set_index(['Device', 'ResultNumber'], inplace=True)
energy_MLModel_6and2.drop(columns=['CPUmj', 'LPMmj', 'DEEPLPMmj', 'TXmj', 'RXmj'], inplace=True)
########################################################################################


# Energy predict 6 and 2
energy_predict_6and2 = energy_predict.copy()
energy_predict_6and2.drop(energy_predict_6and2.loc[( (energy_predict_6and2['ResultName'] == 'regression') )].index, inplace=True)
energy_predict_6and2.drop(energy_predict_6and2.loc[(energy_predict_6and2['ResultNumber'] != '6') & (energy_predict_6and2['ResultNumber'] != '2')].index, inplace=True)
energy_predict_6and2.drop(columns=['Dataset', 'MLModel', 'ResultName', 'SensorsNumber'], inplace=True)
energy_predict_6and2 = energy_predict_6and2.groupby(['Device', 'ResultNumber'])[['CPUmj', 'LPMmj', 'DEEPLPMmj', 'TXmj', 'RXmj']].mean().reset_index().copy()
energy_predict_6and2.loc[:,'TXmj'] = 0
list_sum = ['CPUmj', 'LPMmj', 'DEEPLPMmj', 'TXmj', 'RXmj']
energy_predict_6and2.loc[:, 'FinalmJ'] = energy_predict_6and2.loc[:, list_sum].sum(axis=1)
energy_predict_6and2.set_index(['Device', 'ResultNumber'], inplace=True)
energy_predict_6and2.drop(columns=['CPUmj', 'LPMmj', 'DEEPLPMmj', 'TXmj', 'RXmj'], inplace=True)
########################################################################################


# Energy idle
energy_idle_devices = energy_idle.round(0).copy()
energy_idle_devices['FinalmJ'] = 0.0
energy_idle_devices['LPMmj'] = energy_idle_devices['LPMmj'] + energy_idle_devices['DEEPLPMmj']
energy_idle_devices.loc[:, 'DEEPLPMmj'] = 0
list_exptot_sum = ['CPUmj', 'TXmj', 'RXmj', 'LPMmj']
energy_idle_devices['FinalmJ'] = energy_idle_devices[list_exptot_sum].sum(axis=1)
energy_idle_devices.drop(columns=['CPUmj', 'LPMmj', 'DEEPLPMmj', 'TXmj', 'RXmj'], inplace=True)
# Energia fornecida por 2 pilhas AA:
#(2.85Ah ∗ 1.5V ∗ 3600secs. ∗ 2) = 30780 Joules
# a energia FinalmJ está em milijoule, por isso divide por 1000
aa_bat = 30780
#Lifetime in days without FedSensor
energy_idle_devices = energy_idle_devices.sort_values(by=['Device'])
energy_idle_devices.set_index('Device', inplace=True)
########################################################################################


prediction_interval = [2, 10, 30, 60, 240] #2 e 60 segundos
global_model_update_interval = 1 #horas

energy_6and2_total = pd.DataFrame()
idle_lifetime = pd.DataFrame()


for index, inference_interval in enumerate(prediction_interval):
    idle_lifetime[str(prediction_interval[index])] = ((aa_bat / (energy_idle_devices /1000) + 1).round(0)).astype(int)
    temp = energy_idle_devices + \
                    ( (24/global_model_update_interval) *  energy_MLModel_6and2 ) + \
                    ( (86400/inference_interval) *  energy_predict_6and2 )
    temp.rename(columns={"FinalmJ":str(inference_interval)}, inplace=True)
    energy_6and2_total = pd.concat([energy_6and2_total,temp], axis = 1)
    bat_graph_dataset = (aa_bat / (energy_6and2_total / 1000)).round(0)


graphs.plot_lines_lifetime_comparison_6and2(bat_graph_dataset, idle_lifetime)



print('Percentual médio de diminuição da vida útil: \n',((1 - (bat_graph_dataset / idle_lifetime)) * 100).round(2))
