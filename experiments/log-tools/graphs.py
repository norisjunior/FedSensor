# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 17:46:02 2022

@author: noris
"""

import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from math import log10, ceil
import logging
logging.getLogger('matplotlib.font_manager').disabled = True
import pandas as pd
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset








def plot_circular_barplot(device, sensorsNumber, labels, data):

    # device = 'Sensortag'
    # sensorsNumber = '2'
    # labels = ['RX', 'CPU', 'LPM', 'TX']
    # data = [8.165831273915817, 3.334559849330357, 0.09106993383290815, 0.01]
    formattedData = ["%.3f mJ" % value for value in data]

    #number of data points
    n = len(data)
    #find max value for full ring
    # k = 10 ** int(log10(max(data)))
    # m = k * (1 + max(data) // k)
    # max(data) = máximo do Remote: 9.72125
    k = 10 ** int(log10(9.8))
    m = k * (1 + 9.8 // k)

    #radius of donut chart
    r = 1.5
    #calculate width of each ring
    w = r / n

    #create colors along a chosen colormap
    # colors = ("#3E99F4", "#ffb14e", "#9d02d7", "#FE7972") #escolhida #ef767a ou F29195 ou FD6749 ou FD785D ou FD655D ou FE7972 ou F45B69
    colors = ("#37A3BE", "#FF7075", "#FFB100", "#FDCA9B")

    #create figure, axis
    fig, ax = plt.subplots(figsize=(7,5))

    title = device + ' com ' + sensorsNumber + ' sensores'

    fig.suptitle(title, size=16, y=1.05)

    ax.axis("equal")

    #create rings of donut chart
    for i in range(n):
        #hide labels in segments with textprops: alpha = 0 - transparent, alpha = 1 - visible
        innerring, _ = ax.pie([m - data[i], data[i]],
                              radius = r - i * w,
                              startangle = 90,
                              labels = ["", str(labels[i] + "\n" + formattedData[i])],
                              labeldistance = 1 - 1 / (1.3 * (n - i)),
                              textprops = dict(color = "#494545", alpha = 1, ha = "left", va = "bottom", fontweight = "bold"),
                              colors = ["white", colors[i]])

        plt.setp(innerring, width = w, edgecolor = "#EAEAEA", linewidth = 0.5)

    plt.savefig('figs/Cap6-Sec1-circularbarplot_' + device + "_" + sensorsNumber, bbox_inches='tight')
    plt.show()

    return



def plot_stacked_barplot_MLModel(data):
    sns.set()

    font_color = '#525252'
    csfont = {'fontname':'Roboto Condensed'} # title font
    hfont = {'fontname':'Calibri'} # main font
    # colors = ['#f47e7a', '#b71f5c', '#621237', '#dbbaa7']
    colors = ["#37A3BE", "#FF7075", "#FFB100", "#FDCA9B"]


    # data = graph_dataset

    data.rename(columns = {'Device':'Dispositivo', 'CPUmj':'CPU', 'LPMmj':'LPM', 'TXmj':'TX', 'RXmj':'RX'}, inplace = True)

    data.index = data.Dispositivo

    ax = data.loc[:, ['RX', 'CPU', 'LPM', 'TX']].plot.bar(align='center', stacked=True, figsize=(7, 5), color=colors, rot=0)

    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_color('gray')
    ax.spines['bottom'].set_color('gray')

    # ax.set_ylim((0,32))

    ax.set_facecolor('white')

    ax.grid(axis='y', linestyle=':', linewidth='1.2', color='gray')

    ax.set_ylabel("energia (mJ)")


    # plt.tight_layout()

    if data['MLModel'].unique().item() != '32102':
        title = 'Consumo de energia dataset ' \
            + str(data['Dataset'].unique().item()) \
            + '\nmodelo ' + str(data['ModelName'].unique().item()) \
            + ' (' + str(data['ResultNumber'].unique().item()) \
            + ' ' + str(data['ModelNameResultType'].unique().item()) \
            + ') usando ' + str(data['SensorsNumber'].unique().item()) + ' sensores'
        filename = 'Cap6-Sec2-' \
            + str(data['Dataset'].unique().item()) + '_' \
            + str(data['ModelName'].unique().item()) + '_' \
            + str(data['ResultNumber'].unique().item()) + '_' \
            + str(data['SensorsNumber'].unique().item()) + '_sensores'
    else:
        title = 'Consumo de energia dataset ' \
            + str(data['Dataset'].unique().item()) \
            + '\nmodelo ' + str(data['ModelName'].unique().item()) \
            + ' usando ' + str(data['SensorsNumber'].unique().item()) + ' sensores'
        filename = 'Cap6-Sec2-' \
            + str(data['Dataset'].unique().item()) + '_' \
            + str(data['ModelName'].unique().item()) + '_' \
            + str(data['SensorsNumber'].unique().item()) + '_sensores'


    title = plt.title(title, pad=10, fontsize=14, color=font_color, **csfont)
    title.set_position([.5, 1.02])

    # Adjust the subplot so that the title would fit
    # plt.subplots_adjust(top=0.8, left=0.26)


    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(14)
    plt.xticks(color=font_color, **hfont)
    plt.yticks(range(0,31,5),color=font_color, **hfont)


    plt.savefig('figs/' + filename, bbox_inches='tight')
    plt.show()

    return




def plot_lines_predict(data):

    star_command_blue = '#0C79AC'
    spanish_orange = '#E36414'
    skobeloff_green = '#006166'

    # data = graph_dataset

    data.rename(columns = {'Device':'Dispositivo', 'CPUmj':'CPU', 'LPMmj':'LPM', 'TXmj':'TX', 'RXmj':'RX'}, inplace = True)

    # data.index = data.Device

    title = 'Consumo de energia para inferência (processamento) ' \
        + '\nmodelo ' + str(data['ModelName'].unique().item()) \
        + ' usando ' + str(data['SensorsNumber'].unique().item()) + ' sensores'


    sns.set_style("white", {"axes.axis": "y", "axes.linewidth": 1.25, "axes.edgecolor": "black", "grid.color": ".6", "grid.linestyle": ":", "figsize":"(7,5)"})
    # sns.set_style("white", {"axes.axis": "y", "axes.linewidth": 1, "axes.markersize": 25, "axes.edgecolor": "black", "grid.color": ".6", "grid.linestyle": ":", "figsize":"(7,5)"})

    # sns.set_palette(['#f47e7a', '#b71f5c', '#621237'])
    colors = [star_command_blue, spanish_orange, skobeloff_green]
    sns.set_palette(colors)
    sns.despine(left=False)

    g = sns.relplot(data=data, x="ResultNumber", y="CPU", hue="Dispositivo", kind="line", style="Dispositivo", markers=('o', '^', 's'))
    # g = sns.relplot(data=data, x="ResultNumber", y="CPU", hue="Dispositivo", kind="line", style="Dispositivo", markers=('o', '^', 's'), size=50)
    g.fig.suptitle(title, y=1.07)
    g.set(xlabel=str(data['ModelNameResultType'].unique().item()))
    g.set(ylabel="energia (mJ)")
    g.set(ylim=((0,3.5)))

    filename = 'Cap6-Sec3-' \
            + str(data['Dataset'].unique().item()) + '_' \
            + str(data['ModelName'].unique().item()) + '_' \
            + str(data['SensorsNumber'].unique().item()) + '_sensores'

    plt.savefig('figs/' + filename, bbox_inches='tight')

    plt.show()

    return




def plot_barplot_predict(data):
    sns.set()

    font_color = '#525252'
    csfont = {'fontname':'Roboto Condensed'} # title font
    hfont = {'fontname':'Calibri'} # main font
    # colors = ['#f47e7a', '#b71f5c', '#621237', '#dbbaa7']
    # colors = ["#37A3BE", "#FF7075", "#FFB100", "#FDCA9B"]
    colors = ["#FF7075"]


    # data = graph_dataset

    data.rename(columns = {'Device':'Dispositivo', 'CPUmj':'CPU', 'LPMmj':'LPM', 'TXmj':'TX', 'RXmj':'RX'}, inplace = True)

    data.index = data.Dispositivo

    ax = data.loc[:, ['CPU']].plot.bar(align='center', stacked=False, figsize=(7, 5), color=colors, rot=0)

    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_color('gray')
    ax.spines['bottom'].set_color('gray')

    ax.set_ylim((0,1.25))

    ax.set_facecolor('white')

    ax.grid(axis='y', linestyle=':', linewidth='1.2', color='gray')

    ax.set_ylabel("energia (mJ)")


    plt.tight_layout()

    if data['MLModel'].unique().item() != '32102':
        title = 'Consumo de energia dataset ' \
            + str(data['Dataset'].unique().item()) \
            + '\nmodelo ' + str(data['ModelName'].unique().item()) \
            + ' (' + str(data['ResultNumber'].unique().item()) \
            + ' ' + str(data['ModelNameResultType'].unique().item()) \
            + ') usando ' + str(data['SensorsNumber'].unique().item()) + ' sensores'
    else:
        title = 'Consumo de energia dataset ' \
            + str(data['Dataset'].unique().item()) \
            + '\nmodelo ' + str(data['ModelName'].unique().item()) \
            + ' usando ' + str(data['SensorsNumber'].unique().item()) + ' sensores'


    title = plt.title(title, pad=10, fontsize=14, color=font_color, **csfont)
    title.set_position([.5, 1.02])

    # Adjust the subplot so that the title would fit
    # plt.subplots_adjust(top=0.8, left=0.26)

    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(12)
    plt.xticks(color=font_color, **hfont)
    # plt.yticks(range(0,2,1),color=font_color, **hfont)

    filename = 'Cap6-Sec3-' \
            + str(data['Dataset'].unique().item()) + '_' \
            + str(data['ModelName'].unique().item()) + '_' \
            + str(data['SensorsNumber'].unique().item()) + '_sensores'

    plt.savefig('figs/' + filename, bbox_inches='tight')

    plt.show()

    return







def plot_bar_line_batlife(predict_interval, energy_data, bat_data, atualiza_sensores):
    sns.set()

    font_color = '#525252'
    csfont = {'fontname':'Roboto Condensed'} # title font
    hfont = {'fontname':'Calibri'} # main font
    colors = ['#f47e7a', '#b71f5c', '#621237']


    #https://www.youtube.com/watch?v=-8PpZQZ60s8
    #annotation example: https://queirozf.com/entries/add-labels-and-text-to-matplotlib-plots-annotation-examples
    def linelabel():
        for device in range(len(energy_data.index.values.tolist())):
            line = ax2.lines[device]
            for x_value, y_value in zip(line.get_xdata(), line.get_ydata()):
                space = 4
                va = 'bottom'
                ax2.annotate(
                    y_value,
                    (x_value, y_value),
                    xytext=(0, space),
                    textcoords="offset points",
                    ha='center',
                    va=va,
                    fontsize=10,
                    backgroundcolor="white",
                    bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.5),
                    color="black")


    # predict_interval = 2
    # bat_data = bat_lifetime_FedSensor
    # energy_data = energy_daily_FedSensor
    # energy_data.iloc[:, 1:] = energy_data.iloc[:, 1:].apply(lambda x: x / 1000) #não dividir
    # atualiza_sensores = True

    # x ticks
    global_model_update_interval = energy_data.columns

    # axis 1 (bars): values
    sensortag_energy = np.round(np.array(energy_data.loc[ (energy_data.index == "Sensortag") ]), decimals=3).tolist()[0]
    remote_energy = np.round(np.array(energy_data.loc[ (energy_data.index == "Remote") ]), decimals=3).tolist()[0]
    CC1352P1_energy = np.round(np.array(energy_data.loc[ (energy_data.index == "CC1352P1") ]), decimals=3).tolist()[0]

    # axis 2 (lines): values
    sensortag_lifetime = bat_data.loc[ (bat_data.index == "Sensortag") ].values[0]
    remote_lifetime = bat_data.loc[ (bat_data.index == "Remote") ].values[0]
    CC1352P1_lifetime = bat_data.loc[ (bat_data.index == "CC1352P1") ].values[0]

    # bar size and position
    width = 0.25
    x_CC1352P1  = [x - width for x in range(len(CC1352P1_energy))]
    x_remote    = [x for x in range(len(remote_energy))]
    x_sensortag = [x + width for x in range(len(sensortag_energy))]

    # creating the plot area
    fig,ax = plt.subplots(figsize=(8,5))

    # axis1 configs:
    # ax.spines['left'].set_visible(True)
    # ax.spines['bottom'].set_visible(True)
    # ax.spines['left'].set_color('grey')
    # ax.spines['bottom'].set_color('grey')
    ax.set_ylim((0,100000))
    # ax.set_ylim((0,300000))
    ax.set_facecolor('white')

    # axis 2 configs:
    ax2 = ax.twinx()
    ax2.grid(visible=False)
    ax2.set_ylim((0,140))
    # ax2.set_ylim((0,220))
    ax2.spines['left'].set_visible(True)
    ax2.spines['bottom'].set_visible(True)
    ax2.spines['left'].set_color('#525252')
    ax2.spines['bottom'].set_color('#525252')

    # creating bars
    rect1 = ax.bar(x_CC1352P1,  CC1352P1_energy,  width, label = 'CC1352P1',  color = colors[0])
    rect2 = ax.bar(x_remote,    remote_energy,    width, label = 'Remote',    color = colors[1])
    rect3 = ax.bar(x_sensortag, sensortag_energy, width, label = 'Sensortag', color = colors[2])

    # creating lines
    line1 = ax2.plot(global_model_update_interval, CC1352P1_lifetime,  label='CC1352P1',  linestyle = '-',  color = colors[0], marker = 'o')
    line2 = ax2.plot(global_model_update_interval, remote_lifetime,    label='Remote',    linestyle = '--', color = colors[1], marker = '^')
    line3 = ax2.plot(global_model_update_interval, sensortag_lifetime, label='Sensortag', linestyle = ':',  color = colors[2], marker = 's')

    # labelling lines
    linelabel()

    # title
    if atualiza_sensores:
        msg_title_update = 'com atualização dos sensores'
    else:
        msg_title_update = 'sem atualização dos sensores'
    msg_title = 'Vida útil da bateria e consumo diário de energia para\ntomada de decisão a cada ' + str(predict_interval) + ' segundos ' + msg_title_update
    ax.set_title(msg_title, pad=15, fontsize=14, color=font_color, **csfont)

    # x and y axis labels:
    ax.set_xlabel('Frequência de atualização do modelo global (hora)')
    ax.set_ylabel('consumo diário de energia (mJ)')
    ax2.set_ylabel('vida útil da bateria (dias)')

    # legends
    ax.legend(loc='lower left', bbox_to_anchor=(0., -0.25), fancybox=False, shadow=False, ncol=3)
    ax2.legend(loc='lower left', bbox_to_anchor=(0., -0.35), fancybox=False, shadow=False, ncol=3)

    # plot!
    plt.show()

    return







def plot_model_vs_predict(energy_data, predict_interval):
    # https://stackoverflow.com/questions/69242928/how-to-create-grouped-and-stacked-bars
    # df = pd.melt(model_vs_pred, id_vars=['Device', 'Interval'], value_vars=['global_update', 'predict'])
    # perc_df = pd.melt(model_vs_pred, id_vars=['Device', 'Interval'], value_vars=['global_update_perc', 'predict_perc'])
    # bat_df = pd.melt(model_vs_pred, id_vars=['Device', 'Interval'], value_vars=['bat_lifetime'])
    # predict_interval = 10

    # energy_data = model_vs_pred

    df = pd.melt(energy_data, id_vars=['Device', 'Interval'], value_vars=['global_update', 'predict'])
    perc_df = pd.melt(energy_data, id_vars=['Device', 'Interval'], value_vars=['global_update_perc', 'predict_perc'])
    bat_df = pd.melt(energy_data, id_vars=['Device', 'Interval'], value_vars=['bat_lifetime'])


    sns.set()

    font_color = '#525252'
    csfont = {'fontname':'Roboto Condensed'} # title font
    hfont = {'fontname':'Calibri'} # main font
    colors = ['#f47e7a', '#b71f5c', '#621237']



    labels=df['Interval'].drop_duplicates()  # set the dates as labels
    x0 = np.arange(len(labels))  # create an array of values for the ticks that can perform arithmetic with width (w)

    # create the data groups with a dict comprehension and groupby
    data = {''.join(k): v for k, v in df.groupby(['Device', 'variable'])}
    perc = {''.join(k): v for k, v in perc_df.groupby(['Device', 'variable'])}

    # build the plots
    device = df.Device.unique()
    stacks = len(device)  # how many stacks in each group for a tick location
    variable = df.variable.unique()

    # set the width
    w = 0.3

    # this needs to be adjusted based on the number of stacks; each location needs to be split into the proper number of locations
    x1 = [x0 - w, x0, x0 + w]
    # fill_colors = ["#37A3BE", "#FF7075", "#FFB100"]

    maximum_yellow_red = "#FFCD70"
    baby_blue = "#7CCEF4"

    beige = "#D5DECE"
    cotton_candy = "#FFC2D7"

    laurel_green = "#C0CEB6"
    blue_bell = "#A0A0CF"

    melon = "#F6C5B7"

    pacific_blue = "#41ADC8"
    honey_yellow = "#FFB100"

    fill_colors = [pacific_blue, honey_yellow]

    border_colors = plt.cm.Paired.colors

    fig, ax = plt.subplots(figsize=(14,5))
    ax.set_facecolor('white')

    for x, dev in zip(x1, device):
        bottom = 0
        for index, var in enumerate(variable):
            height = data[f'{dev}{var}'].value.to_numpy()
            perc_value = perc[f'{dev}{var}_perc'].value.to_numpy().round(1)
            perc_value = [f'{x}%' for x in perc_value]
            graph = ax.bar(x=x, height=height, width=w, bottom=bottom, edgecolor="none", color=fill_colors[index], clip_on=False)
            ax.bar_label(graph, labels=perc_value, label_type='center', fontsize=8, padding=0, rotation=0, fontname="Arial Nova Cond Light", clip_on=False)
            for artist in graph:
                print(artist)

            # ax.bar_label(vbar, labels=dev, padding=8, color='b', fontsize=14)
            # ax.set_xticks(height,dev)
            # ax.xticks(x1[0], ['CC1352P1', 'CC1352P1', 'CC1352P1', 'CC1352P1', 'CC1352P1', 'CC1352P1', 'CC1352P1', 'CC1352P1'])
            # ax.xticks(x1[1], ['Remote', 'Remote', 'Remote', 'Remote', 'Remote', 'Remote', 'Remote', 'Remote'])
            # ax.xticks(x1[2], ['Sensortag', 'Sensortag', 'Sensortag', 'Sensortag', 'Sensortag', 'Sensortag', 'Sensortag', 'Sensortag'])
            bottom += height

    ax.legend(['atualização do modelo global', 'inferência'],
              title="Energia",
              fontsize=10,
              loc='lower left',
              bbox_to_anchor=(0.325, -0.45),
              fancybox=False,
              shadow=False,
              ncol=3)

    ylim_max = int(ceil(max(energy_data.loc[( (energy_data['Device'] == 'Remote') ), 'FinalmJ']) / 1000))*1000
    ylim_min = ylim_max * -0.01

    # ax.set_ylim((-100,10000))
    ax.set_ylim((ylim_min, ylim_max))

    # fig.subplots_adjust(bottom=0.9)


    # Mostra nomes dos dispositivos no eixo 'x'
    # x_label_devices_distance = -1450
    x_label_devices_distance = ylim_min * 14.5
    for index, positions in enumerate(x1):
        for device_position in positions:
            # print(index,device_position,device[index])
            ax.text((device_position-w/3)+0.05, x_label_devices_distance, device[index], fontsize = 8, rotation=90)

    ax.set_xticks(x0, fontsize=10)
    _ = ax.set_xticklabels([('%f' % x).rstrip('0').rstrip('.') for x in labels])
    ax.tick_params(axis='x', pad=40)
    # ax.set_xlabel('Frequência de atualização do modelo global (hora)', labelpad=12)


    # axis 2 configs:
    ax2 = ax.twinx()
    ax2.grid(visible=False)
    ax2.set_ylim((0,150))
    ax2.spines['left'].set_visible(True)
    ax2.spines['bottom'].set_visible(True)
    ax2.spines['left'].set_color('#525252')
    ax2.spines['bottom'].set_color('#525252')

    # x1[0][0]
    # x1['device']['interval']
    # --> x1[0][0] ==> CC1352P1 0.125
    # --> x1[0][1] ==> CC1352P1 0.5
    # --> x1[2][7] ==> Sensortag 24

    # plot lines
    space = 4
    va = 'bottom'
    for index, interval in enumerate(labels):
        ax2.plot(
            [x1[0][index], x1[1][index], x1[2][index]],
            bat_df.loc[(bat_df['Interval'] == interval),['value']].T.values.tolist()[0],
            color='C3', marker='o', mec='k'
            )
        line = ax2.lines[index]
        for x_value, y_value in zip(line.get_xdata(), line.get_ydata()):
            ax2.annotate(
                y_value,
                (x_value, y_value),
                xytext=(0, space),
                textcoords="offset points",
                ha='center',
                va=va,
                fontsize=10,
                backgroundcolor="white",
                bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.5),
                color="black")

    # title
    msg_title = 'Comparativo entre a energia média gasta para atualização do modelo global\ne a energia média gasta para inferência (frequência das inferências = ' + str(predict_interval) + ' segundos)'
    ax.set_title(msg_title, pad=10, fontsize=14, color=font_color, **csfont)

    # x and y axis labels:
    ax.set_xlabel('Frequência de atualização do modelo global (hora)', labelpad=12)
    ax.set_ylabel('consumo diário de energia (mJ)')
    ax2.set_ylabel('vida útil da bateria (dias)')

    # fig.texts.append(ax.texts.pop())

    filename = 'Cap6-Sec4-global_update_vs_predict_' + str(predict_interval) + '_secs'

    plt.savefig('figs/' + filename, bbox_inches='tight')

    plt.show()

    return











def plot_lines_lifetime_comparison_predict(data, idle_data, interval):

    # data = bat_lifetime_FedSensor
    # idle_data = idle_lifetime
    # interval = 0.25


    # data.index = data.Device


    # data = data.reset_index().rename(columns={model_vs_pred.index.name:'Device'})

    # df = pd.melt(data, id_vars=['Device'], value_vars=['2', '10', '30', '60', '240'])

    # sns.set()

    # title = 'Comparativo da vida útil de bateria dos dispositivos\n(considerando atualização do modelo global a cada 1 hora)'

    # sns.set_style("white", {"axes.axis": "y", "axes.linewidth": 1, "axes.edgecolor": "black", "grid.color": ".6", "grid.linestyle": ":"})
    # sns.set_palette(['#f47e7a', '#b71f5c', '#621237'])
    # sns.despine(left=False)

    # g = sns.relplot(data=data, x=df.variable, y=df.value, hue=df.Device, kind="line", style=df.Device, markers=('o', '^', 's'), height=5, aspect=1.5)
    # # g.figsize(10,15)
    # g.fig.suptitle(title, y=1.07)
    # g.set(xlabel="Frequência da tomada de decisão (segundos)")
    # g.set(ylabel="vida útil da bateria (dias)")
    # g.set(ylim=((60,140)))

    # plt.show()



    sns.set()

    font_color = '#525252'
    csfont = {'fontname':'Roboto Condensed'} # title font
    hfont = {'fontname':'Calibri'} # main font
    # colors = ['#f47e7a', '#b71f5c', '#621237']
    star_command_blue = '#0C79AC'
    spanish_orange = '#E36414'
    skobeloff_green = '#006166'

    colors = [star_command_blue, spanish_orange, skobeloff_green]



    # creating the plot area
    fig,ax = plt.subplots(figsize=(15,7))

    ax.grid(visible=False)
    ax.set_ylim((70,140))
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_color('#525252')
    ax.spines['bottom'].set_color('#525252')

    #creating lines of idle data from the devices
    line4 = ax.plot(idle_data.columns, idle_data.loc[idle_data.index == 'CC1352P1'].values[0].tolist(),  label='CC1352P1 idle',  linestyle = '--',  linewidth=1, color = 'darkgray', marker = 'o', markersize=8)
    line5 = ax.plot(idle_data.columns, idle_data.loc[idle_data.index == 'Remote'].values[0].tolist(),  label='Remote idle',  linestyle = '--',  linewidth=1, color = 'darkgray', marker = '^', markersize=8)
    line6 = ax.plot(idle_data.columns, idle_data.loc[idle_data.index == 'Sensortag'].values[0].tolist(),  label='Sensortag idle',  linestyle = '--', linewidth=1, color = 'darkgray', marker = 's', markersize=8)

    # creating lines of data
    line1 = ax.plot(data.columns, data.loc[data.index == 'CC1352P1'].values[0].tolist(),  label='CC1352P1',  linestyle = '-', linewidth=3, color = colors[0], marker = 'o', markersize=10)
    line2 = ax.plot(data.columns, data.loc[data.index == 'Remote'].values[0].tolist(),  label='Remote',  linestyle = '--', linewidth=3, color = colors[1], marker = '^', markersize=10)
    line3 = ax.plot(data.columns, data.loc[data.index == 'Sensortag'].values[0].tolist(),  label='Sensortag',  linestyle = ':', linewidth=3, color = colors[2], marker = 's', markersize=10)


    # title
    msg_title = 'Comparativo da vida útil média de bateria dos dispositivos com a utilização do FedSensor\n(considerando atualização do modelo global a cada ' + str(interval) + ' hora)'
    # é média porque se faz uma média com o uso de 9 e de 2 sensores
    # e nessa média entra o experimento MOTOR que consome menos energia, puxando pra cima a média
    ax.set_title(msg_title, pad=15, fontsize=16, color=font_color, **csfont)

    # x and y axis labels:
    ax.set_xlabel('Frequência da tomada de decisão (segundos)', fontsize=14, labelpad=12)
    ax.set_ylabel('vida útil da bateria (dias)', fontsize=14)

    # legends
    # ax.legend(labels=['CC1352P1', 'Remote', 'Sensortag', 'CC1352P1 idle', 'Remote idle', 'Sensortag idle'], title='Dispositivo', loc='lower left', bbox_to_anchor=(0.11, -0.275), fancybox=False, shadow=False, ncol=6)
    ax.legend(title='Dispositivo', loc='lower left', bbox_to_anchor=(0.11, -0.275), fancybox=False, shadow=False, ncol=6)

    # ticks size
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(14)
        label.color=font_color
    plt.xticks(color=font_color, fontsize=14)

    # x ticks space


    filename = 'Cap6-Sec4-lifetime_comparison_predict_' + str(interval) + '.png'

    plt.savefig('figs/' + filename, bbox_inches='tight')

    # plot!
    plt.show()


    return










def plot_lines_lifetime_comparison_update(data, idle_data, interval):

    # data = bat_lifetime_FedSensor
    # idle_data = idle_lifetime
    # interval = 10

    sns.set()

    font_color = '#525252'
    csfont = {'fontname':'Roboto Condensed'} # title font
    hfont = {'fontname':'Calibri'} # main font
    # colors = ['#f47e7a', '#b71f5c', '#621237']
    star_command_blue = '#0C79AC'
    spanish_orange = '#E36414'
    skobeloff_green = '#006166'

    colors = [star_command_blue, spanish_orange, skobeloff_green]



    # creating the plot area
    fig,ax = plt.subplots(figsize=(15,7))

    ax.grid(visible=False)
    ax.set_ylim((70,140))
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_color('#525252')
    ax.spines['bottom'].set_color('#525252')

    #creating lines of idle data from the vices
    line4 = ax.plot(idle_data.columns, idle_data.loc[idle_data.index == 'CC1352P1'].values[0].tolist(),  label='CC1352P1 idle',  linestyle = '--',  linewidth=1, color = 'darkgray', marker = 'o', markersize=8)
    line5 = ax.plot(idle_data.columns, idle_data.loc[idle_data.index == 'Remote'].values[0].tolist(),  label='Remote idle',  linestyle = '--',  linewidth=1, color = 'darkgray', marker = '^', markersize=8)
    line6 = ax.plot(idle_data.columns, idle_data.loc[idle_data.index == 'Sensortag'].values[0].tolist(),  label='Sensortag idle',  linestyle = '--', linewidth=1, color = 'darkgray', marker = 's', markersize=8)

    # creating lines of data
    line1 = ax.plot(data.columns, data.loc[data.index == 'CC1352P1'].values[0].tolist(),  label='CC1352P1',  linestyle = '-', linewidth=3, color = colors[0], marker = 'o', markersize=10)
    line2 = ax.plot(data.columns, data.loc[data.index == 'Remote'].values[0].tolist(),  label='Remote',  linestyle = '--', linewidth=3, color = colors[1], marker = '^', markersize=10)
    line3 = ax.plot(data.columns, data.loc[data.index == 'Sensortag'].values[0].tolist(),  label='Sensortag',  linestyle = ':', linewidth=3, color = colors[2], marker = 's', markersize=10)


    # title
    msg_title = 'Comparativo da vida útil média de bateria dos dispositivos com a utilização do FedSensor\n(considerando inferência a cada ' + str(interval) + ' segundos)'
    ax.set_title(msg_title, pad=15, fontsize=16, color=font_color, **csfont)

    # x and y axis labels:
    ax.set_xlabel('Frequência de atualização do modelo global (horas)', fontsize=14, labelpad=12)
    ax.set_ylabel('vida útil da bateria (dias)', fontsize=14)

    # legends
    # ax.legend(labels=['CC1352P1', 'Remote', 'Sensortag', 'CC1352P1 idle', 'Remote idle', 'Sensortag idle'], title='Dispositivo', loc='lower left', bbox_to_anchor=(0.11, -0.275), fancybox=False, shadow=False, ncol=6)
    ax.legend(title='Dispositivo', loc='lower left', bbox_to_anchor=(0.11, -0.275), fancybox=False, shadow=False, ncol=6)

    # ticks size
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(14)
        label.color=font_color
    plt.xticks(color=font_color, fontsize=14)

    # x ticks space

    filename = 'Cap6-Sec4-lifetime_comparison_update_' + str(interval) + '.png'

    plt.savefig('figs/' + filename, bbox_inches='tight')


    # plot!
    plt.show()


    return






def plot_lines_federated_training_loss_3_participants(loss_data):

    # loss_data = graph_dataset
    dataset = loss_data['Dataset'].unique()
    algorithm = loss_data['MLModel'].unique()
    sensorsNumber = loss_data['n_sensors'].unique().item()
    n_classes = loss_data['n_classes'].unique().item()
    ##### anomaly_detection = loss_data['anomaly_detection'].unique()
    training_rounds = [index+1 for index, item in enumerate(loss_data['result'].values[0])]

    # if (anomaly_detection == "No"): anomaly_detection = "sem"

    if (dataset == 'AQI'): dataset_name = 'IQAr'

    if (algorithm == 'logreg'):
        MLmodel = 'regressão logística'
    else:
        print("Não é possível gerar gráfico para k-means nem regressão linear")
        return
    # data.index = data.Device


    sns.set()

    font_color = '#525252'
    csfont = {'fontname':'Roboto Condensed'} # title font
    hfont = {'fontname':'Calibri'} # main font
    # colors = ['#f47e7a', '#b71f5c', '#621237']
    colors = plt.cm.Paired.colors



    # creating the plot area
    fig,ax = plt.subplots(figsize=(15,7))

    ax.grid(visible=False)
    ax.set_ylim((0,0.6))
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_color('#525252')
    ax.spines['bottom'].set_color('#525252')

    no_anomaly_data = [value[1] for index, value in enumerate(loss_data.loc[(loss_data['anomaly_detection'] == 'No'), ['result']].values[0][0])]
    IForest_data = [value[1] for index, value in enumerate(loss_data.loc[(loss_data['anomaly_detection'] == 'IForest'), ['result']].values[0][0])]
    ECOD_data = [value[1] for index, value in enumerate(loss_data.loc[(loss_data['anomaly_detection'] == 'ECOD'), ['result']].values[0][0])]

    # creating lines of data
    line1 = ax.plot(training_rounds, no_anomaly_data,  label='Com anomalias',  linestyle = '-', linewidth=3, color = colors[0], marker = 'o', markersize=10)
    line2 = ax.plot(training_rounds, IForest_data,  label='Detecção de anomalias com IForest',  linestyle = '--', linewidth=3, color = colors[1], marker = '^', markersize=10)
    line3 = ax.plot(training_rounds, ECOD_data,  label='Detecção de anomalias com ECOD',  linestyle = ':', linewidth=3, color = colors[2], marker = 's', markersize=10)


    msg_title = 'Função de custo federada com modelo ' + MLmodel + ' cenário ' + dataset_name + ' com ' + str(sensorsNumber) + ' sensores e ' + str(n_classes) + ' classes'
    ax.set_title(msg_title, pad=15, fontsize=16, color=font_color, **csfont)

    # x and y axis labels:
    ax.set_xlabel('rodadas de treinamento federado', fontsize=14, labelpad=12)
    ax.set_ylabel('custo federado (resultado da função de perda)', fontsize=14)

    # legends
    # ax.legend(labels=['CC1352P1', 'Remote', 'Sensortag', 'CC1352P1 idle', 'Remote idle', 'Sensortag idle'], title='Dispositivo', loc='lower left', bbox_to_anchor=(0.11, -0.275), fancybox=False, shadow=False, ncol=6)
    ax.legend()

    # ticks size
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(14)
        label.color=font_color
    plt.xticks(color=font_color, fontsize=14)

    # x ticks space

    # plot!
    plt.show()


    return









def plot_lines_federated_training_loss_5_participants(loss_data):


    # loss_data = graph_dataset
    dataset = loss_data['Dataset'].unique()
    algorithm = loss_data['MLModel'].unique()
    sensorsNumber = loss_data['n_sensors'].unique().item()
    n_classes = loss_data['n_classes'].unique().item()
    training_rounds = [index+1 for index, item in enumerate(loss_data['loss_result'].values[0])]

    if (dataset == 'AQI'): dataset_name = 'IQAr'

    if (algorithm == 'logreg'):
        MLmodel = 'regressão logística'
    else:
        print("Não é possível gerar gráfico para k-means nem regressão linear")
        return

    #São 7 linhas:
        # 5 com anomalia (todos)
        # 5 sem anomalia (todos) usando IForest
        # 5 sem anomalia (todos) usando ECOD
        # 4 com anomalia e 1 sem anomalia usando IForest
        # 4 com anomalia e 1 sem anomalia usando ECOD
        # 2 com anomalia e 2 sem anomalia usando IForest
        # 2 com anomalia e 2 sem anomalia usando ECOD

    all_with_anomaly = [value[1] for index, value in enumerate(loss_data.loc[(loss_data['anomaly_detector'] == ''), ['loss_result']].values[0][0])]
    all_IForest_detector = [value[1] for index, value in enumerate(loss_data.loc[((loss_data['anomaly_detector'] == 'IForest') & (loss_data['participants_with_detection'] == '5')), ['loss_result']].values[0][0])]
    all_ECOD_detector = [value[1] for index, value in enumerate(loss_data.loc[((loss_data['anomaly_detector'] == 'ECOD') & (loss_data['participants_with_detection'] == '5')), ['loss_result']].values[0][0])]
    anomaly_4_IForest_1 = [value[1] for index, value in enumerate(loss_data.loc[((loss_data['anomaly_detector'] == 'IForest') & (loss_data['participants_with_detection'] == '1')), ['loss_result']].values[0][0])]
    anomaly_4_ECOD_1 = [value[1] for index, value in enumerate(loss_data.loc[((loss_data['anomaly_detector'] == 'ECOD') & (loss_data['participants_with_detection'] == '1')), ['loss_result']].values[0][0])]
    anomaly_2_IForest_3 = [value[1] for index, value in enumerate(loss_data.loc[((loss_data['anomaly_detector'] == 'IForest') & (loss_data['participants_with_detection'] == '3')), ['loss_result']].values[0][0])]
    anomaly_2_ECOD_3 = [value[1] for index, value in enumerate(loss_data.loc[((loss_data['anomaly_detector'] == 'ECOD') & (loss_data['participants_with_detection'] == '3')), ['loss_result']].values[0][0])]


    sns.set()

    font_color = '#525252'
    csfont = {'fontname':'Roboto Condensed'} # title font
    hfont = {'fontname':'Calibri'} # main font
    # colors = ['#f47e7a', '#b71f5c', '#621237']
    # colors = plt.cm.twilight.colors



    # creating the plot area
    fig,ax = plt.subplots(figsize=(15,7))

    ax.grid(visible=False)
    ax.set_ylim((0,3))
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_color('#525252')
    ax.spines['bottom'].set_color('#525252')


    dark_orange = '#FB8B24'
    mellow_apricot = '#FCB573'
    spanish_orange = '#E36414'
    midnight_green_eagle_green = '#0B3A46' #0B3A46 ou #0F4C5C
    ruby_red = '#9A031E'
    whatever_red = '#E65F5C'
    very_dark_green = '#004346'
    saphire_blue = '#0B6C99'
    shadow = '#937B63'
    gunmetal = '#2E3138'
    silver_chalice = '#AEADAE'
    ash_gray = '#B9B7A7'
    peach_crayola = '#FDCA9B'
    pink_lace = '#FBD6FF'
    light_gray = '#D9D3D7'
    black_shadows = '#BDB3B9'

    #vermelho, verde, laranja
    colors = [black_shadows, mellow_apricot, ruby_red, very_dark_green, saphire_blue]
    # creating lines of data
    line1 = ax.plot(training_rounds, anomaly_4_IForest_1,  label='Somente 1 participante com detecção de anomalias usando IForest',  linestyle = 'solid', linewidth=1, color = colors[0], marker = 'X', markersize=7)
    line1 = ax.plot(training_rounds, anomaly_4_ECOD_1,  label='Somente 1 participante com detecção de anomalias usando ECOD',  linestyle = 'solid', linewidth=1, color = colors[0], marker = 'D', markersize=7)
    line2 = ax.plot(training_rounds, anomaly_2_IForest_3,  label='Somente 3 participantes com detecção de anomalias usando IForest',  linestyle = 'solid', linewidth=1, color = colors[1], marker = 'P', markersize=7)
    line2 = ax.plot(training_rounds, anomaly_2_ECOD_3,  label='Somente 3 participantes com detecção de anomalias usando ECOD',  linestyle = 'solid', linewidth=1, color = colors[1], marker = 'p', markersize=7)
    line3 = ax.plot(training_rounds, all_with_anomaly,  label='Todos os participantes com anomalia',  linestyle = 'dashed', linewidth=1.5, color = colors[2], marker = 'o', markersize=8)
    line4 = ax.plot(training_rounds, all_IForest_detector,  label='Detecção de anomalias em todos os participantes usando IForest',  linestyle = 'solid', linewidth=2, color = colors[3], marker = '^', markersize=8)
    line5 = ax.plot(training_rounds, all_ECOD_detector,  label='Detecção de anomalias em todos os participantes usando ECOD',  linestyle = 'solid', linewidth=2, color = colors[4], marker = 's', markersize=8)



    # line1 = ax.plot(training_rounds, no_anomaly_data,  label='Com anomalias',  linestyle = 'solid', linewidth=3, color = colors[0], marker = 'o', markersize=10)
    # line2 = ax.plot(training_rounds, IForest_data,  label='Detecção de anomalias com IForest',  linestyle = 'dashed', linewidth=3, color = colors[1], marker = '^', markersize=10)
    # line3 = ax.plot(training_rounds, ECOD_data,  label='Detecção de anomalias com ECOD',  linestyle = 'dotted', linewidth=3, color = colors[2], marker = 's', markersize=10)


    # msg_title = 'Função de custo federada no FedSensor contendo 5 participantes\ncom modelo ' + MLmodel + ' cenário ' + dataset_name + ' com ' + str(sensorsNumber) + ' sensores e ' + str(n_classes) + ' classes'
    # msg_title = 'Função de custo federada no FedSensor contendo 5 participantes\nusando modelo global ' + MLmodel + ' com ' + str(sensorsNumber) + ' sensores e ' + str(n_classes) + ' classes'
    msg_title = 'Função de custo federada no FedSensor contendo 5 participantes\nusando modelo global ' + MLmodel + ' com ' + str(sensorsNumber) + ' sensores'
    ax.set_title(msg_title, pad=15, fontsize=16, color=font_color, **csfont)

    # x and y axis labels:
    ax.set_xlabel('rodadas de treinamento federado', fontsize=14, labelpad=12)
    ax.set_ylabel('custo federado (resultado da função de perda)', fontsize=14)


    #https://towardsdatascience.com/mastering-inset-axes-in-matplotlib-458d2fdfd0c0
    # Create an inset axis in the bottom right corner
    axin = ax.inset_axes([0.65, 0.45, 0.325, 0.2])
    # Plot the data on the inset axis and zoom in on the important part
    axin.plot(training_rounds, anomaly_4_IForest_1,  linestyle = 'solid', linewidth=1, color = colors[0], marker = 'X', markersize=7)
    axin.plot(training_rounds, anomaly_4_ECOD_1, linestyle = 'solid', linewidth=1, color = colors[0], marker = 'D', markersize=7)
    axin.plot(training_rounds, anomaly_2_IForest_3, linestyle = 'solid', linewidth=1, color = colors[1], marker = 'P', markersize=7)
    axin.plot(training_rounds, anomaly_2_ECOD_3, linestyle = 'solid', linewidth=1, color = colors[1], marker = 'p', markersize=7)
    axin.plot(training_rounds, all_with_anomaly, linestyle = 'dashed', linewidth=1.5, color = colors[2], marker = 'o', markersize=8)
    axin.plot(training_rounds, all_IForest_detector, linestyle = 'solid', linewidth=2, color = colors[3], marker = '^', markersize=8)
    axin.plot(training_rounds, all_ECOD_detector, linestyle = 'solid', linewidth=2, color = colors[4], marker = 's', markersize=8)
    axin.set_xlim(40, 51)
    axin.set_ylim(0.15, 0.3)
    # Add the lines to indicate where the inset axis is coming from
    ax.indicate_inset_zoom(axin)




    # legends
    # ax.legend(labels=['CC1352P1', 'Remote', 'Sensortag', 'CC1352P1 idle', 'Remote idle', 'Sensortag idle'], title='Dispositivo', loc='lower left', bbox_to_anchor=(0.11, -0.275), fancybox=False, shadow=False, ncol=6)
    ax.legend()

    # ticks size
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(14)
        label.color=font_color
    plt.xticks(color=font_color, fontsize=14)

    # x ticks space

    # plot!
    plt.show()


    return






def plot_rasp_energy_total(energy_data):
    # https://stackoverflow.com/questions/69242928/how-to-create-grouped-and-stacked-bars
    # df = pd.melt(model_vs_pred, id_vars=['Device', 'Interval'], value_vars=['global_update', 'predict'])
    # perc_df = pd.melt(model_vs_pred, id_vars=['Device', 'Interval'], value_vars=['global_update_perc', 'predict_perc'])
    # bat_df = pd.melt(model_vs_pred, id_vars=['Device', 'Interval'], value_vars=['bat_lifetime'])

    # energy_data = energy_daily_Raspi.copy()

    energy_data[['idle_mJ', 'local_training_mJ', 'rec_from_manager_mJ', 'trans_to_manager_mJ', 'trans_to_device_mJ']] = energy_data[['idle_mJ', 'local_training_mJ', 'rec_from_manager_mJ', 'trans_to_manager_mJ', 'trans_to_device_mJ']]/1000

    energy_data.reset_index(inplace=True)
    energy_data['devices_and_sensors'] = energy_data['n_devices'].astype(str) +" disp.-"+ energy_data["n_sensors"].astype(str) + " sensores"
    energy_data.drop(columns=['n_devices', 'n_sensors'], inplace=True)
    energy_data['global_model_update_interval'] = energy_data['global_model_update_interval'].astype(str)


    df = pd.melt(energy_data, id_vars=['devices_and_sensors', 'global_model_update_interval'], value_vars=['idle_mJ', 'local_training_mJ', 'rec_from_manager_mJ', 'trans_to_manager_mJ', 'trans_to_device_mJ'])
    # perc_df = pd.melt(energy_data, id_vars=['Device', 'Interval'], value_vars=['global_update_perc', 'predict_perc'])
    # bat_df = pd.melt(energy_data, id_vars=['Device', 'Interval'], value_vars=['bat_lifetime'])






    sns.set()

    font_color = '#525252'
    csfont = {'fontname':'Roboto Condensed'} # title font
    hfont = {'fontname':'Calibri'} # main font
    colors = ['#f47e7a', '#b71f5c', '#621237']



    labels=list(df['global_model_update_interval'].drop_duplicates())  # set the dates as labels
    x0 = np.arange(len(labels))  # create an array of values for the ticks that can perform arithmetic with width (w)

    # create the data groups with a dict comprehension and groupby
    data = {''.join(k): v for k, v in df.groupby(['devices_and_sensors', 'variable'])}
    # perc = {''.join(k): v for k, v in perc_df.groupby(['Device', 'variable'])}

    # build the plots
    devices = list(df['devices_and_sensors'].unique())
    stacks = len(devices)  # how many stacks in each group for a tick location
    variable = df.variable.unique()

    # set the width
    w = 0.1

    # this needs to be adjusted based on the number of stacks; each location needs to be split into the proper number of locations
    # x1 = [x0 - w, x0, x0 + w] #aqui são 8 não 3
    x1 = [x0 - (3.5*w), x0 - (2.5*w), x0 - (1.5*w), x0 - w/2, x0 + w/2, x0 + (1.5*w), x0 + (2.5*w), x0 + (3.5*w)]
    fill_colors = plt.cm.Paired.colors
    border_colors = plt.cm.Paired.colors

    fig, ax = plt.subplots(figsize=(16,3))
    ax.set_facecolor('white')
    ax.grid(visible=False)
    ax.set_ylim((0,3))
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_color('#525252')
    ax.spines['bottom'].set_color('#525252')


    for x, dev in zip(x1, devices):
        bottom = 0
        for index, var in enumerate(variable):
            height = data[f'{dev}{var}'].value.to_numpy()
            # perc_value = perc[f'{dev}{var}_perc'].value.to_numpy().round(1)
            # perc_value = [f'{x}%' for x in perc_value]
            graph = ax.bar(x=x, height=height, width=w, bottom=bottom, color=fill_colors[index], edgecolor="none", clip_on=False)
            # ax.bar_label(graph, labels=perc_value, label_type='center', fontsize=8, padding=0, rotation=0, fontname="Arial Nova Cond Light", clip_on=True)
            for artist in graph:
                print(artist)

            # ax.bar_label(vbar, labels=dev, padding=8, color='b', fontsize=14)
            # ax.set_xticks(height,dev)
            # ax.xticks(x1[0], ['CC1352P1', 'CC1352P1', 'CC1352P1', 'CC1352P1', 'CC1352P1', 'CC1352P1', 'CC1352P1', 'CC1352P1'])
            # ax.xticks(x1[1], ['Remote', 'Remote', 'Remote', 'Remote', 'Remote', 'Remote', 'Remote', 'Remote'])
            # ax.xticks(x1[2], ['Sensortag', 'Sensortag', 'Sensortag', 'Sensortag', 'Sensortag', 'Sensortag', 'Sensortag', 'Sensortag'])
            bottom += height

    ax.set_ylim((0,130000))

    # plt.ylim(bottom = 80000)

    ax.legend(['inativo', 'treinamento local', 'recebimento do modelo global', 'transmissão do modelo local p/ nuvem', 'transmissão do modelo local p/ disp.'],
              title="Energia",
              fontsize=10,
              loc='lower left',
              bbox_to_anchor=(0.05, -1.),
              fancybox=False,
              shadow=False,
              ncol=5)


    # fig.subplots_adjust(bottom=0.9)

    for index, positions in enumerate(x1):
        for device_position in positions:
            # print(index,device_position,variable[index])
            ax.text((device_position-(w/1.5))+0.05, -59000, devices[index], fontsize = 8, rotation=90)

    ax.set_xticks(x0, fontsize=10)
    _ = ax.set_xticklabels([x.rstrip('0').rstrip('.') for x in labels]) #não tá entendendo vir como string
    ax.tick_params(axis='x', pad=80)
    # ax.set_xlabel('Frequência de atualização do modelo global (hora)', labelpad=12)


    # title
    msg_title = 'Comparativo entre a energia diária gasta para o treinamento federado com a aplicação da\nseleção de variáveis, considerando diferentes números de dispositivos e de seus sensores (variáveis do modelo)'
    ax.set_title(msg_title, pad=10, fontsize=14, color=font_color, **csfont)

    # x and y axis labels:
    ax.set_xlabel('Frequência de atualização do modelo global (hora)', labelpad=12)
    ax.set_ylabel('consumo diário de energia (J)')

    plt.show()


    return









def plot_bar_energy_9and2(predict_interval, energy_data):

    # predict_interval = 60
    # energy_data = graph_dataset

    sns.set()

    font_color = '#525252'
    csfont = {'fontname':'Roboto Condensed'} # title font
    hfont = {'fontname':'Calibri'} # main font
    colors = ['#f47e7a', '#b71f5c', '#621237']

    # x ticks
    global_model_update_interval = list(energy_data.columns)

    # axis 1 (bars): values
    sensortag_energy = np.round(energy_data.loc['Sensortag'].to_numpy()[0], decimals=3).tolist()
    remote_energy = np.round(energy_data.loc['Remote'].to_numpy()[0], decimals=3).tolist()
    CC1352P1_energy = np.round(energy_data.loc['CC1352P1'].to_numpy()[0], decimals=3).tolist()


    # bar size and position
    width = 0.25
    x_CC1352P1  = [x - width for x in range(len(CC1352P1_energy))]
    x_remote    = [x for x in range(len(remote_energy))]
    x_sensortag = [x + width for x in range(len(sensortag_energy))]

    tick_centers = [x for x in range(len(remote_energy))]

    # creating the plot area
    fig,ax = plt.subplots(figsize=(8,5))

    # axis1 configs:
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_color('grey')
    ax.spines['bottom'].set_color('grey')
    ax.set_ylim((0,600000))
    ax.set_facecolor('white')

    # creating bars
    rect1 = ax.bar(x_CC1352P1,  CC1352P1_energy,  width, label = 'CC1352P1',  color = colors[0])
    rect2 = ax.bar(x_remote,    remote_energy,    width, label = 'Remote',    color = colors[1])
    rect3 = ax.bar(x_sensortag, sensortag_energy, width, label = 'Sensortag', color = colors[2])


    plt.xticks(ticks = tick_centers, labels = global_model_update_interval)

    # ax.xaxis.set_ticks(global_model_update_interval)


    # ax.set_xticklabels(global_model_update_interval)

    msg_title = 'Consumo diário de energia para tomada de decisão a cada ' + str(predict_interval) + ' segundos\n' + 'sem seleção de variáveis (' + str(energy_data.index.get_level_values('SensorsNumber').unique().values.item()) + ' sensores)'
    ax.set_title(msg_title, pad=15, fontsize=14, color=font_color, **csfont)

    # x and y axis labels:
    ax.set_xlabel('Frequência de atualização do modelo global (hora)')
    ax.set_ylabel('consumo diário de energia (mJ)')

    # legends
    ax.legend(loc='lower left', bbox_to_anchor=(0., -0.25), fancybox=False, shadow=False, ncol=3)

    # plot!
    plt.show()

    return









def plot_bar_energy_6and2(predict_interval, energy_data):

    # predict_interval = 60
    # energy_data = graph_dataset

    sns.set()

    font_color = '#525252'
    csfont = {'fontname':'Roboto Condensed'} # title font
    hfont = {'fontname':'Calibri'} # main font
    colors = ['#f47e7a', '#b71f5c', '#621237']

    # x ticks
    global_model_update_interval = list(energy_data.columns)

    # axis 1 (bars): values
    sensortag_energy = np.round(energy_data.loc['Sensortag'].to_numpy()[0], decimals=3).tolist()
    remote_energy = np.round(energy_data.loc['Remote'].to_numpy()[0], decimals=3).tolist()
    CC1352P1_energy = np.round(energy_data.loc['CC1352P1'].to_numpy()[0], decimals=3).tolist()


    # bar size and position
    width = 0.25
    x_CC1352P1  = [x - width for x in range(len(CC1352P1_energy))]
    x_remote    = [x for x in range(len(remote_energy))]
    x_sensortag = [x + width for x in range(len(sensortag_energy))]

    tick_centers = [x for x in range(len(remote_energy))]

    # creating the plot area
    fig,ax = plt.subplots(figsize=(8,5))

    # axis1 configs:
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_color('grey')
    ax.spines['bottom'].set_color('grey')
    ax.set_ylim((0,600000))
    ax.set_facecolor('white')

    # creating bars
    rect1 = ax.bar(x_CC1352P1,  CC1352P1_energy,  width, label = 'CC1352P1',  color = colors[0])
    rect2 = ax.bar(x_remote,    remote_energy,    width, label = 'Remote',    color = colors[1])
    rect3 = ax.bar(x_sensortag, sensortag_energy, width, label = 'Sensortag', color = colors[2])


    plt.xticks(ticks = tick_centers, labels = global_model_update_interval)

    # ax.xaxis.set_ticks(global_model_update_interval)


    # ax.set_xticklabels(global_model_update_interval)

    msg_title = 'Consumo diário de energia para tomada de decisão a cada ' + str(predict_interval) + ' segundos\n' + 'usando (' + str(energy_data.index.get_level_values('ResultNumber').unique().values.item()) + ' classes/grupos)'
    ax.set_title(msg_title, pad=15, fontsize=14, color=font_color, **csfont)

    # x and y axis labels:
    ax.set_xlabel('Frequência de atualização do modelo global (hora)')
    ax.set_ylabel('consumo diário de energia (mJ)')

    # legends
    ax.legend(loc='lower left', bbox_to_anchor=(0., -0.25), fancybox=False, shadow=False, ncol=3)

    # plot!
    plt.show()

    return
















def plot_lines_lifetime_comparison_9and2(data, idle_data):

    # data = bat_graph_dataset
    # idle_data = idle_lifetime


    dark_orange = '#FB8B24'
    mellow_apricot = '#FCB573'
    spanish_orange = '#E36414'
    midnight_green_eagle_green = '#0B3A46' #0B3A46 ou #0F4C5C
    ruby_red = '#9A031E'
    whatever_red = '#E65F5C'
    very_dark_green = '#004346'
    saphire_blue = '#0B6C99'
    shadow = '#937B63'
    gunmetal = '#2E3138'
    silver_chalice = '#AEADAE'
    ash_gray = '#B9B7A7'
    peach_crayola = '#FDCA9B'
    pink_lace = '#FBD6FF'
    light_gray = '#D9D3D7'
    black_shadows = '#BDB3B9'
    skobeloff_green = '#006166'
    star_command_blue = '#0C79AC'


    #vermelho, verde, laranja
    # colors = [black_shadows, mellow_apricot, ruby_red, very_dark_green, saphire_blue]


    sns.set()

    font_color = '#525252'
    csfont = {'fontname':'Roboto Condensed'} # title font
    hfont = {'fontname':'Calibri'} # main font
    # colors = ['#f47e7a', '#b71f5c', '#621237']
    colors = [star_command_blue, spanish_orange, skobeloff_green]


    # creating the plot area
    fig,ax = plt.subplots(figsize=(15,7))

    ax.grid(visible=False)
    ax.set_ylim((50,130))
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_color('#525252')
    ax.spines['bottom'].set_color('#525252')

    #creating lines of idle data from the devices
    linea = ax.plot(idle_data.columns, idle_data.loc[idle_data.index == 'CC1352P1'].values[0].tolist(),  label='CC1352P1',  linestyle = '--',  linewidth=1, color = 'darkgray', marker = 'o', markersize=8)
    lineb = ax.plot(idle_data.columns, idle_data.loc[idle_data.index == 'Remote'].values[0].tolist(),  label='Remote',  linestyle = '--',  linewidth=1, color = 'darkgray', marker = '^', markersize=8)
    linec = ax.plot(idle_data.columns, idle_data.loc[idle_data.index == 'Sensortag'].values[0].tolist(),  label='Sensortag',  linestyle = '--', linewidth=1, color = 'darkgray', marker = 's', markersize=8)

    # creating lines of data
    line_cc1352p1_2 = ax.plot(data.columns, data.loc['CC1352P1'].to_numpy()[0].astype(int),  label='CC1352P1 2 sensores',  linestyle = '-', linewidth=3, color = colors[0], marker = 'o', markersize=10)
    line_cc1352p1_9 = ax.plot(data.columns, data.loc['CC1352P1'].to_numpy()[1].astype(int),  label='CC1352P1 9 sensores',  linestyle = ':', linewidth=3, color = colors[0], marker = 'o', markersize=10)

    line_remote_2 = ax.plot(data.columns, data.loc['Remote'].to_numpy()[0].astype(int),  label='Remote 2 sensores',  linestyle = '-', linewidth=3, color = colors[1], marker = '^', markersize=10)
    line_remote_9 = ax.plot(data.columns, data.loc['Remote'].to_numpy()[1].astype(int),  label='Remote 9 sensores',  linestyle = ':', linewidth=3, color = colors[1], marker = '^', markersize=10)

    line_sensortag_2 = ax.plot(data.columns, data.loc['Sensortag'].to_numpy()[0].astype(int),  label='Sensortag 2 sensores',  linestyle = '-', linewidth=3, color = colors[2], marker = 's', markersize=10)
    line_sensortag_9 = ax.plot(data.columns, data.loc['Sensortag'].to_numpy()[1].astype(int),  label='Sensortag 9 sensores',  linestyle = ':', linewidth=3, color = colors[2], marker = 's', markersize=10)


    # title
    msg_title = 'Comparativo da vida útil de bateria dos dispositivos com a utilização da seleção de variáveis\n(comparando número de sensores/variáveis do modelo) em diferentes frequências\nde tomada de decisão (considerando atualização do modelo global a cada 1 hora)'
    ax.set_title(msg_title, pad=15, fontsize=16, color=font_color, **csfont)

    # x and y axis labels:
    ax.set_xlabel('Frequência da tomada de decisão (segundos)', fontsize=14, labelpad=11)
    ax.set_ylabel('vida útil da bateria (dias)', fontsize=14)

    # legends
    ax.legend(title='Dispositivo', loc='lower left', bbox_to_anchor=(0.03, -0.3), fancybox=False, shadow=False, ncol=5)

    # ticks size
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(14)
        label.color=font_color
    plt.xticks(color=font_color, fontsize=14)

    # save the plot

    filename = 'Cap6-Sec5-feature_selection_features_9and2.png'

    plt.savefig('figs/' + filename, bbox_inches='tight')


    # plot!
    plt.show()




    return












def plot_lines_lifetime_comparison_6and2(data, idle_data):

    # data = bat_graph_dataset
    # idle_data = idle_lifetime


    dark_orange = '#FB8B24'
    mellow_apricot = '#FCB573'
    spanish_orange = '#E36414'
    midnight_green_eagle_green = '#0B3A46' #0B3A46 ou #0F4C5C
    ruby_red = '#9A031E'
    whatever_red = '#E65F5C'
    very_dark_green = '#004346'
    saphire_blue = '#0B6C99'
    shadow = '#937B63'
    gunmetal = '#2E3138'
    silver_chalice = '#AEADAE'
    ash_gray = '#B9B7A7'
    peach_crayola = '#FDCA9B'
    pink_lace = '#FBD6FF'
    light_gray = '#D9D3D7'
    black_shadows = '#BDB3B9'
    skobeloff_green = '#006166'
    star_command_blue = '#0C79AC'


    #vermelho, verde, laranja
    # colors = [black_shadows, mellow_apricot, ruby_red, very_dark_green, saphire_blue]


    sns.set()

    font_color = '#525252'
    csfont = {'fontname':'Roboto Condensed'} # title font
    hfont = {'fontname':'Calibri'} # main font
    # colors = ['#f47e7a', '#b71f5c', '#621237']
    colors = [star_command_blue, spanish_orange, skobeloff_green]

    # creating the plot area
    fig,ax = plt.subplots(figsize=(15,7))

    ax.grid(visible=False)
    ax.set_ylim((45,130))
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_color('#525252')
    ax.spines['bottom'].set_color('#525252')

    #creating lines of idle data from the devices
    linea = ax.plot(idle_data.columns, idle_data.loc[idle_data.index == 'CC1352P1'].values[0].tolist(),  label='CC1352P1',  linestyle = '--',  linewidth=1, color = 'darkgray', marker = 'o', markersize=8)
    lineb = ax.plot(idle_data.columns, idle_data.loc[idle_data.index == 'Remote'].values[0].tolist(),  label='Remote',  linestyle = '--',  linewidth=1, color = 'darkgray', marker = '^', markersize=8)
    linec = ax.plot(idle_data.columns, idle_data.loc[idle_data.index == 'Sensortag'].values[0].tolist(),  label='Sensortag',  linestyle = '--', linewidth=1, color = 'darkgray', marker = 's', markersize=8)

    # creating lines of data
    line_cc1352p1_2 = ax.plot(data.columns, data.loc['CC1352P1'].to_numpy()[0].astype(int),  label='CC1352P1 2 classes',  linestyle = '-', linewidth=3, color = colors[0], marker = 'o', markersize=10)
    line_cc1352p1_9 = ax.plot(data.columns, data.loc['CC1352P1'].to_numpy()[1].astype(int),  label='CC1352P1 6 classes',  linestyle = ':', linewidth=3, color = colors[0], marker = 'o', markersize=10)

    line_remote_2 = ax.plot(data.columns, data.loc['Remote'].to_numpy()[0].astype(int),  label='Remote 2 classes',  linestyle = '-', linewidth=3, color = colors[1], marker = '^', markersize=10)
    line_remote_9 = ax.plot(data.columns, data.loc['Remote'].to_numpy()[1].astype(int),  label='Remote 6 classes',  linestyle = ':', linewidth=3, color = colors[1], marker = '^', markersize=10)

    line_sensortag_2 = ax.plot(data.columns, data.loc['Sensortag'].to_numpy()[0].astype(int),  label='Sensortag 2 classes',  linestyle = '-', linewidth=3, color = colors[2], marker = 's', markersize=10)
    line_sensortag_9 = ax.plot(data.columns, data.loc['Sensortag'].to_numpy()[1].astype(int),  label='Sensortag 6 classes',  linestyle = ':', linewidth=3, color = colors[2], marker = 's', markersize=10)


    # title
    msg_title = 'Comparativo da vida útil de bateria dos dispositivos com a utilização da seleção de variáveis\n(comparando número de classes resultantes do desfecho do modelo) em diferentes frequências\nde tomada de decisão (considerando atualização do modelo global a cada 1 hora)'
    ax.set_title(msg_title, pad=15, fontsize=16, color=font_color, **csfont)

    # x and y axis labels:
    ax.set_xlabel('Frequência da tomada de decisão (segundos)', fontsize=14, labelpad=11)
    ax.set_ylabel('vida útil da bateria (dias)', fontsize=14)

    # legends
    ax.legend(title='Dispositivo', loc='lower left', bbox_to_anchor=(0.03, -0.3), fancybox=False, shadow=False, ncol=5)

    # ticks size
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(14)
        label.color=font_color
    plt.xticks(color=font_color, fontsize=14)

    filename = 'Cap6-Sec5-feature_selection_outcome_6and2.png'

    plt.savefig('figs/' + filename, bbox_inches='tight')

    # plot!
    plt.show()




    return
