#Imports
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
sns.set() # Setting seaborn as default style even if use only matplotlib
plt.rcParams.update({"font.family": "Roboto Condensed"})
from statistics import variance
import graphs

#Load data
df_anomaly_detection = pd.read_pickle("anomaly_result_5_participants.pkl")

df_anomaly_detection['category'] = df_anomaly_detection['Dataset']+df_anomaly_detection['MLModel']+df_anomaly_detection['n_sensors']+df_anomaly_detection['n_classes']

df_anomaly_detection.drop(df_anomaly_detection.loc[(df_anomaly_detection['Dataset'] == 'MOTOR')].index, inplace=True)

df_anomaly_detection.drop(df_anomaly_detection.loc[(df_anomaly_detection['n_classes'] != '2')].index, inplace = True)


for i in df_anomaly_detection.index:
    df_anomaly_detection.loc[df_anomaly_detection.index==i, 'variance_last_10_rounds'] = variance([values[1] for values in df_anomaly_detection.loc[df_anomaly_detection.index==i, 'loss_result'].values.tolist()[0][39:50]])


for category in df_anomaly_detection['category'].unique():
    graph_dataset = df_anomaly_detection.loc[(df_anomaly_detection['category'] == category)]
    graphs.plot_lines_federated_training_loss_5_participants(graph_dataset)
