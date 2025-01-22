import csv
import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.integrate import simps
from scipy.stats import linregress
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import scienceplots
from matplotlib.lines import Line2D 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.interpolate import griddata
import paper_figure_processor_alts as alts
from sklearn.preprocessing import MinMaxScaler
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import MaxNLocator

# comment for nice plot styling
matplotlib.style.use(['science', 'nature'])

# INSERT PATH TO DATA FILES HERE 
spring_data_file = ''
speed_data_file =  ''
diameter_data_file = '' 
# ------------

pressures = [0,0.5 * 100000, 0.75 * 100000,1.00 * 100000,1.25 * 100000,1.50 * 100000] # convert to pa
diameters = [16,20,24,28]
materials = [63,70,82]
rs = [d / 2000 for d in diameters]  # convert to m 
L = 50 / 1000
try:
    spring_data = pd.read_csv(spring_data_file)
except:
    print("FILE PATHS ABOVE LIKELY INCORRECTLY SET")
    raise SystemExit

spring_data['Diameter'] = spring_data['Diameter'].replace(25,24)
speed_data = pd.read_csv(speed_data_file)
speed_data['Diameter'] = speed_data['Diameter'].replace(25,24)
diameter_data = pd.read_csv(diameter_data_file)
diameter_data['Diameter'] = diameter_data['Diameter'].replace(25,24)
#diameter_data.loc[diameter_data['Pressure'] == 150, 'Delta Vol'] += 14
#diameter_data.loc[diameter_data['Pressure'] == 100, 'Delta Vol'] *= 2
#diameter_data.to_csv('C:\\Users\\mackj\\OneDrive - University of Edinburgh\\PhD\\SpiderPump\\Diameter tests\\output4.csv' , index=False)
speed_data_avg = speed_data.groupby(speed_data.index // 3).mean()
spring_data_avg = spring_data.groupby(spring_data.index // 3).mean()
diameter_data_avg = diameter_data.groupby(diameter_data.index // 3).mean()
mask = (speed_data_avg['Diameter'] == 28) & (speed_data_avg['Pressure'] == 150)
speed_data_avg = speed_data_avg[~mask]
spring_data['Buckle Force'] = spring_data['Buckle Force'].multiply(10)
spring_data['Unbuckle Stiffness'] = spring_data['Buckle Slope'].multiply(1 / L **2)
speed_data_avg['Angular Rate'] = 90 / speed_data_avg['Full dt']

# Figure 1 
def figure_1():
    diameters = [16,20,24,28]
    materials = [60,70,82]
    material_plot_keys = ['o','s','^']
    material_plot_fc =  ['#1aaf6c','#429bf4','#d42cea']
    material_plot_labels = ['60','70','82']
    lw_ratios = [0.4,0.5,0.625,0.7]
    filtered_data = spring_data[(spring_data['Pressure'] == -50) & (spring_data['Diameter'].isin(diameters))]
    x_err = [0.1,0.1,0.1]
    fig, axs = plt.subplots(figsize=(8,5),dpi=300)
    legend_handles = []
    for i, diameter in enumerate(diameters):
        for j, material in enumerate(materials):
            # Filter data for each diameter and mat
            subset = filtered_data[(filtered_data['Diameter'] == diameter) & (filtered_data['Material'] == material)]
            avg_force = subset['Peak Load Force'].mean() * 10
            subset['Peak Load Force Adjusted'] = subset['Peak Load Force'] * 10
            axs.errorbar(diameters[i], avg_force,xerr=0.01, yerr=subset['Peak Load Force Adjusted'].sem(), marker = material_plot_keys[j], markersize = 8, alpha = 0.7,label = material_plot_labels[j],linestyle = '', color = material_plot_fc[j])
            if i == 0:
                legend_handles.append(Line2D([0], [0], color=material_plot_fc[j], marker=material_plot_keys[j], linestyle='-', label=f'{material}A'))
    axs.set_title('Peak Retraction Force for Evacuated Actuator')
    axs.legend(handles=legend_handles, title='Material', fontsize=8)
    axs.set_xlabel('Diameter (mm)')
    axs.set_ylabel('Peak Force (N)')
    plt.show()


#def compute_sem(data, key):
    #mean = data[key].mean()
    #std_dev = data[key].std(ddof=1)
    #return data[key].sem()


def figure_3_pannel_cont(data_in,key,units=''):
    # Create fine grid
    data = data_in.copy()
    data['Pressure'] = data['Pressure'] / 100
    df = data[data['Material'] == 60]
    diameter_fine = np.linspace(df['Diameter'].min(), df['Diameter'].max(), 100)
    pressure_fine = np.linspace(df['Pressure'].min(), df['Pressure'].max(), 100)
    diameter_grid, pressure_grid = np.meshgrid(diameter_fine, pressure_fine)
    plt.figure(figsize=(12, 5), dpi=300)
    plt.subplot(1, 3, 1)
    TITLE_SIZE = 14
    LABEL_SIZE = 12
    TICK_SIZE = 10
    # Interpolate work
    work_fine = griddata(
        (df['Diameter'], df['Pressure']), 
        df[key], 
        (diameter_grid, pressure_grid), 
        method='cubic'
    )

    # Filled contour plot
    contourf = plt.contourf(
        diameter_grid, 
        pressure_grid, 
        work_fine, 
        levels=np.linspace(df[key].min(), df[key].max(), 10), 
        cmap='plasma', 
        alpha=0.75
    )
    cbar = plt.colorbar(contourf,  orientation='horizontal', pad =0.1)
    cbar.locator = MaxNLocator(nbins=5)  # Use at most 5 bins
    cbar.set_label(f'{key} {units}', fontsize = 10)
    cbar.formatter = FormatStrFormatter('%.1f')
    cbar.update_ticks()

    # Contour lines
    contour = plt.contour(
        diameter_grid, 
        pressure_grid, 
        work_fine, 
        levels=np.linspace(df[key].min(), df[key].max(), 10), 
        colors='black',
        linewidths=0.5
    )
    plt.clabel(contour, inline=True, fontsize=TICK_SIZE)

    # Add markers
    #plt.scatter(df['Diameter'], df['Pressure'], color='red', edgecolor='black')
    #for i, txt in enumerate(df['Unload Work']):
        #plt.text(df['Diameter'][i], df['Pressure'][i], f'{txt}', fontsize=9, ha='right')

    plt.xlabel('Diameter (mm)', fontsize = LABEL_SIZE)
    plt.ylabel('Pressure (Bar)', fontsize = LABEL_SIZE)
    plt.grid(True)
    plt.title(f'{key} for the 60A Samples', fontsize = TITLE_SIZE)
    plt.legend(fontsize = 10)
    plt.tick_params(axis='both', which='major', labelsize = TICK_SIZE)

    df = data[data['Material'] == 70]
    diameter_fine = np.linspace(df['Diameter'].min(), df['Diameter'].max(), 100)
    pressure_fine = np.linspace(df['Pressure'].min(), df['Pressure'].max(), 100)
    diameter_grid, pressure_grid = np.meshgrid(diameter_fine, pressure_fine)
    plt.subplot(1, 3, 2)
    # Interpolate work
    work_fine = griddata(
        (df['Diameter'], df['Pressure']), 
        df[key], 
        (diameter_grid, pressure_grid), 
        method='cubic'
    )

    # Filled contour plot
    contourf = plt.contourf(
        diameter_grid, 
        pressure_grid, 
        work_fine, 
        levels=np.linspace(df[key].min(), df[key].max(), 10), 
        cmap='plasma', 
        alpha=0.75
    )
    cbar = plt.colorbar(contourf, orientation='horizontal', pad =0.1)
    cbar.locator = MaxNLocator(nbins=5)  # Use at most 5 bins
    cbar.set_label(f'{key} {units}',fontsize=10)
    cbar.formatter = FormatStrFormatter('%.1f')
    cbar.update_ticks()

    # Contour lines
    contour = plt.contour(
        diameter_grid, 
        pressure_grid, 
        work_fine, 
        levels=np.linspace(df[key].min(), df[key].max(), 10), 
        colors='black',
        linewidths=0.5
    )
    plt.clabel(contour, inline=True, fontsize=TICK_SIZE)

    # Add markers
    #plt.scatter(df['Diameter'], df['Pressure'], color='red', edgecolor='black')
    #for i, txt in enumerate(df['Unload Work']):
        #plt.text(df['Diameter'][i], df['Pressure'][i], f'{txt}', fontsize=9, ha='right')

    plt.xlabel('Diameter (mm)', fontsize = LABEL_SIZE)
    plt.ylabel('Pressure (Bar)', fontsize = LABEL_SIZE)
    plt.grid(True)
    plt.title(f'{key} for the 70A Samples', fontsize = TITLE_SIZE)
    plt.legend(fontsize = 10)
    plt.tick_params(axis='both', which='major', labelsize = TICK_SIZE)

    df = data[data['Material'] == 82]
    diameter_fine = np.linspace(df['Diameter'].min(), df['Diameter'].max(), 100)
    pressure_fine = np.linspace(df['Pressure'].min(), df['Pressure'].max(), 100)
    diameter_grid, pressure_grid = np.meshgrid(diameter_fine, pressure_fine)
    plt.subplot(1, 3, 3)
    # Interpolate work
    work_fine = griddata(
        (df['Diameter'], df['Pressure']), 
        df[key], 
        (diameter_grid, pressure_grid), 
        method='cubic'
    )

    # Filled contour plot
    contourf = plt.contourf(
        diameter_grid, 
        pressure_grid, 
        work_fine, 
        levels=np.linspace(df[key].min(), df[key].max(), 10), 
        cmap='plasma', 
        alpha=0.75
    )
    cbar = plt.colorbar(contourf,  orientation='horizontal', pad =0.1)
    cbar.locator = MaxNLocator(nbins=5)  # Use at most 5 bins
    cbar.set_label(f'{key} {units}',fontsize=10)
    cbar.formatter = FormatStrFormatter('%.1f')
    cbar.update_ticks()
    # Contour lines
    contour = plt.contour(
        diameter_grid, 
        pressure_grid, 
        work_fine, 
        levels=np.linspace(df[key].min(), df[key].max(), 10), 
        colors='black',
        linewidths=0.5
    )
    plt.clabel(contour, inline=True, fontsize=TICK_SIZE)

    # Add markers
    #plt.scatter(df['Diameter'], df['Pressure'], color='red', edgecolor='black')
    #for i, txt in enumerate(df['Unload Work']):
        #plt.text(df['Diameter'][i], df['Pressure'][i], f'{txt}', fontsize=9, ha='right')

    plt.xlabel('Diameter (mm)', fontsize = LABEL_SIZE)
    plt.ylabel('Pressure (Bar)', fontsize = LABEL_SIZE)
    plt.grid(True)
    plt.title(f'{key} for the 82A Samples', fontsize = TITLE_SIZE)
    plt.legend(fontsize = 10)
    plt.tick_params(axis='both', which='major', labelsize = TICK_SIZE)
    plt.tight_layout()
    plt.show()


# Figure 4
def figure_4(spring_data):
    df = spring_data[(spring_data['Material'] == 60) & (spring_data['Pressure'] >= 0)]
    plt.figure(figsize=(12, 5))
    # Plot Pressure vs. buckle
    plt.subplot(1, 3, 1)
    df_16 = df[df['Diameter'] == 16]
    plt.plot(df_16['Pressure'], df_16['Buckle Angle'], marker='s', label='16mm', linestyle='--', color='b')
    df_20 = df[df['Diameter'] == 20]
    plt.plot(df_20['Pressure'], df_20['Buckle Angle'], marker='o', label='20mm', linestyle='--', color='b')
    df_25 = df[df['Diameter'] == 25]
    plt.plot(df_25['Pressure'], df_25['Buckle Angle'], marker='^', label = '25mm', linestyle='--', color='b')
    df_28 = df[df['Diameter'] == 28]
    plt.plot(df_28['Pressure'], df_28['Buckle Angle'], marker='x', label = '28mm', linestyle='--', color='b')
    plt.xlabel('Pressure')
    plt.ylabel('Buckle Angle')
    plt.title('60A Pressure vs. Buckle Angle')
    plt.legend()
    plt.grid(True)
    
    df = spring_data[(spring_data['Material'] == 70) & (spring_data['Pressure'] >= 0)]
    plt.subplot(1, 3, 2)
    df_16 = df[df['Diameter'] == 16]
    plt.plot(df_16['Pressure'], df_16['Buckle Angle'], marker='s', label='16mm', linestyle='--', color='r')
    df_20 = df[df['Diameter'] == 20]
    plt.plot(df_20['Pressure'], df_20['Buckle Angle'], marker='o', label='20mm', linestyle='--', color='r')
    df_25 = df[df['Diameter'] == 25]
    plt.plot(df_25['Pressure'], df_25['Buckle Angle'], marker='^', label = '25mm', linestyle='--', color='r')
    df_28 = df[df['Diameter'] == 28]
    plt.plot(df_28['Pressure'], df_28['Buckle Angle'], marker='x', label = '28mm', linestyle='--', color='r')
    plt.xlabel('Pressure')
    plt.ylabel('Buckle Angle')
    plt.title('70A Pressure vs. Buckle Angle')
    plt.legend()
    plt.grid(True)

    df = spring_data[(spring_data['Material'] == 82) & (spring_data['Pressure'] >= 0)]
    plt.subplot(1, 3, 3)
    df_16 = df[df['Diameter'] == 16]
    plt.plot(df_16['Pressure'], df_16['Buckle Angle'], marker='s', label='16mm', linestyle='--', color='g')
    df_20 = df[df['Diameter'] == 20]
    plt.plot(df_20['Pressure'], df_20['Buckle Angle'], marker='o', label='20mm', linestyle='--', color='g')
    df_25 = df[df['Diameter'] == 25]
    plt.plot(df_25['Pressure'], df_25['Buckle Angle'], marker='^', label = '25mm', linestyle='--', color='g')
    df_28 = df[df['Diameter'] == 28]
    plt.plot(df_28['Pressure'], df_28['Buckle Angle'], marker='x', label = '28mm', linestyle='--', color='g')
    plt.xlabel('Pressure')
    plt.ylabel('Buckle Angle')
    plt.title('82A Pressure vs. Buckle Angle')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def total_vol_change(data_in,data_in_raw = None):
    #diameters = [16,20,24,28]
    #pressures = [-50,0,50,75,100,125,150]
    #materials = [60,70,82]
    #mat = []
    #pres = []
    #diam = []
    #sems = []
    #for d in diameters:
        #for p in pressures:
            #for m in materials:
                #df = data_in_raw[(data_in_raw['Diameter'] == d) & (data_in_raw['Pressure'] == p) & (data_in_raw['Material'] == m)]
                #print(df)
                #print(df['Delta Vol'].sem())
                #sems.append(df['Delta Vol'].sem())
                #mat.append(m)
                #pres.append(p)
                #diam.append(d)
    #sem_data = {"Diameter" : diam, "Material" : mat, "Pressure" : pres, "SEM" : sems}
    #df_sems = pd.DataFrame(sem_data)

    data = data_in.copy()
    data['Pressure'] = data['Pressure'] / 100
    colors =  ['#1aaf6c','#429bf4','#d42cea','#E2725B']
    df = data[data['Material'] == 60]
    plt.figure(figsize=(12, 5),dpi=250)
    plt.subplot(1, 3, 1)
    TITLE_FONT = 12
    AXIS_FONT = 12
    XY_FONT = 10
    plt.rcParams['font.size'] = TITLE_FONT
    x_err = 0.05 
    y_err = 0.5 
    df_16 = df[(df['Diameter'] == 16)]
    df_16_pos = df[(df['Diameter'] == 16) & (df['Pressure']>= 0)]
    #df_16_sems_pos = df_sems[(df_sems['Diameter'] == 16) & (df_sems['Pressure']>= 0)]
    vac_vol = df_16.loc[df_16['Pressure'] == -0.5, 'Delta Vol2'].iloc[0]
    df_16_pos['Total V'] = df_16_pos['Delta Vol2'] + vac_vol
    plt.errorbar(df_16_pos['Pressure'], df_16_pos['Total V'], xerr=x_err, yerr=y_err, marker='s', label='16mm', linestyle='--', color=colors[0])
    df_20 = df[(df['Diameter'] == 20)]
    df_20_pos = df[(df['Diameter'] == 20) & (df['Pressure']>= 0)]
    vac_vol = df_20.loc[df_20['Pressure'] == -0.5, 'Delta Vol2'].iloc[0]
    df_20_pos['Total V'] = df_20_pos['Delta Vol2'] + vac_vol
    plt.errorbar(df_20_pos['Pressure'], df_20_pos['Total V'], xerr=x_err, yerr=y_err, marker='o', label='20mm', linestyle='--', color=colors[1])
    df_24 = df[(df['Diameter'] == 24)]
    df_24_pos = df[(df['Diameter'] == 24) & (df['Pressure']>= 0)]
    vac_vol = df_24.loc[df_24['Pressure'] == -0.5, 'Delta Vol2'].iloc[0]
    df_24_pos['Total V'] = df_24_pos['Delta Vol2'] + vac_vol
    plt.errorbar(df_24_pos['Pressure'], df_24_pos['Total V'], xerr=x_err, yerr=y_err, marker='^', label = '24mm', linestyle='--', color=colors[2])
    df_28 = df[(df['Diameter'] == 28)]
    df_28_pos = df[(df['Diameter'] == 28) & (df['Pressure']>= 0)]
    vac_vol = df_28.loc[df_28['Pressure'] == -0.5, 'Delta Vol2'].iloc[0]
    df_28_pos['Total V'] = df_28_pos['Delta Vol2'] + vac_vol
    plt.errorbar(df_28_pos['Pressure'], df_28_pos['Total V'], xerr=x_err, yerr=y_err, marker='x', label = '28mm', linestyle='--', color=colors[3])
    plt.xlabel('Pressure (Bar)', fontsize = AXIS_FONT)
    plt.ylabel('Total Volume Change (mL)', fontsize = AXIS_FONT)
    plt.xticks(fontsize=XY_FONT)
    plt.yticks(fontsize=XY_FONT)
    plt.title('60A Pressure vs. Volume Change')
    plt.legend(title="Diameter", fontsize = 10, title_fontsize = 10)
    plt.grid(True)
    df = data[data['Material'] == 70]
    plt.subplot(1, 3, 2)
    plt.rcParams['font.size'] = TITLE_FONT
    df_16 = df[(df['Diameter'] == 16)]
    df_16_pos = df[(df['Diameter'] == 16) & (df['Pressure']>= 0)]
    vac_vol = df_16.loc[df_16['Pressure'] == -0.5, 'Delta Vol2'].iloc[0]
    df_16_pos['Total V'] = df_16_pos['Delta Vol2'] + vac_vol
    plt.errorbar(df_16_pos['Pressure'], df_16_pos['Total V'], xerr=x_err, yerr=y_err, marker='s', label='16mm', linestyle='--', color=colors[0])
    df_20 = df[(df['Diameter'] == 20)]
    df_20_pos = df[(df['Diameter'] == 20) & (df['Pressure']>= 0)]
    vac_vol = df_20.loc[df_20['Pressure'] == -0.5, 'Delta Vol2'].iloc[0]
    df_20_pos['Total V'] = df_20_pos['Delta Vol2'] + vac_vol
    plt.errorbar(df_20_pos['Pressure'], df_20_pos['Total V'], xerr=x_err, yerr=y_err, marker='o', label='20mm', linestyle='--', color=colors[1])
    df_24 = df[(df['Diameter'] == 24)]
    df_24_pos = df[(df['Diameter'] == 24) & (df['Pressure']>= 0)]
    vac_vol = df_24.loc[df_24['Pressure'] == -0.5, 'Delta Vol2'].iloc[0]
    df_24_pos['Total V'] = df_24_pos['Delta Vol2'] + vac_vol
    plt.errorbar(df_24_pos['Pressure'], df_24_pos['Total V'], xerr=x_err, yerr=y_err, marker='^', label = '24mm', linestyle='--', color=colors[2])
    df_28 = df[(df['Diameter'] == 28)]
    df_28_pos = df[(df['Diameter'] == 28) & (df['Pressure']>= 0)]
    vac_vol = df_28.loc[df_28['Pressure'] == -0.5, 'Delta Vol2'].iloc[0]
    df_28_pos['Total V'] = df_28_pos['Delta Vol2'] + vac_vol
    plt.errorbar(df_28_pos['Pressure'], df_28_pos['Total V'], xerr=x_err, yerr=y_err, marker='x', label = '28mm', linestyle='--', color=colors[3])
    plt.xlabel('Pressure (Bar)', fontsize = AXIS_FONT)
    plt.ylabel('Total Volume Change (mL)', fontsize = AXIS_FONT)
    plt.xticks(fontsize=XY_FONT)
    plt.yticks(fontsize=XY_FONT)
    plt.title('70A Pressure vs. Volume Change')
    plt.legend(title="Diameter", fontsize = 10, title_fontsize = 10)
    plt.grid(True)
    df = data[data['Material'] == 82]
    plt.subplot(1, 3, 3)
    plt.rcParams['font.size'] = TITLE_FONT
    df_16 = df[(df['Diameter'] == 16)]
    df_16_pos = df[(df['Diameter'] == 16) & (df['Pressure']>= 0)]
    vac_vol = df_16.loc[df_16['Pressure'] == -0.5, 'Delta Vol2'].iloc[0]
    df_16_pos['Total V'] = df_16_pos['Delta Vol2'] + vac_vol
    plt.errorbar(df_16_pos['Pressure'], df_16_pos['Total V'], xerr=x_err, yerr=y_err, marker='s', label='16mm', linestyle='--', color=colors[0])
    df_20 = df[(df['Diameter'] == 20)]
    df_20_pos = df[(df['Diameter'] == 20) & (df['Pressure']>= 0)]
    vac_vol = df_20.loc[df_20['Pressure'] == -0.5, 'Delta Vol2'].iloc[0]
    df_20_pos['Total V'] = df_20_pos['Delta Vol2'] + vac_vol
    plt.errorbar(df_20_pos['Pressure'], df_20_pos['Total V'], xerr=x_err, yerr=y_err, marker='o', label='20mm', linestyle='--', color=colors[1])
    df_24 = df[(df['Diameter'] == 24)]
    df_24_pos = df[(df['Diameter'] == 24) & (df['Pressure']>= 0)]
    vac_vol = df_24.loc[df_24['Pressure'] == -0.5, 'Delta Vol2'].iloc[0]
    df_24_pos['Total V'] = df_24_pos['Delta Vol2'] + vac_vol
    plt.errorbar(df_24_pos['Pressure'], df_24_pos['Total V'], xerr=x_err, yerr=y_err, marker='^', label = '24mm', linestyle='--', color=colors[2])
    df_28 = df[(df['Diameter'] == 28)]
    df_28_pos = df[(df['Diameter'] == 28) & (df['Pressure']>= 0)]
    vac_vol = df_28.loc[df_28['Pressure'] == -0.5, 'Delta Vol2'].iloc[0]
    df_28_pos['Total V'] = df_28_pos['Delta Vol2'] + vac_vol
    plt.errorbar(df_28_pos['Pressure'], df_28_pos['Total V'], xerr=x_err, yerr=y_err, marker='x', label = '28mm', linestyle='--', color=colors[3])
    plt.xlabel('Pressure (Bar)', fontsize = AXIS_FONT)
    plt.ylabel('Total Volume Change (mL)', fontsize = AXIS_FONT)
    plt.xticks(fontsize=XY_FONT)
    plt.yticks(fontsize=XY_FONT)
    plt.title('82A Pressure vs. Volume Change')
    plt.legend(title="Diameter", fontsize = 10, title_fontsize = 10)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def figure_4b(bl_16,bl_20,bl_25,bl_28,spring_data_in):
    colors =  ['#1aaf6c','#429bf4','#d42cea']
    spring_data = spring_data_in.copy()
    spring_data['Pressure'] = spring_data['Pressure'] / 100
    spring_data['Buckle Force'] = spring_data['Buckle Force'] * 10
    df = spring_data[(spring_data['Material'] == 60) & (spring_data['Pressure'] >= 0)]
    plt.figure(figsize=(12, 5))
    # Plot Pressure vs. buckle
    plt.subplot(1, 3, 1)
    df_16 = df[df['Diameter'] == 16]
    plt.plot(df_16['Pressure'], df_16['Buckle Force'], marker='s', label='16mm', linestyle='', color=colors[0])
    plt.plot(df['Pressure'].unique(), bl_16,label = '16mm Diameter Theoretical Collapse Force', linestyle= '-', color = colors[0], alpha = 1)
    df_20 = df[df['Diameter'] == 20]
    plt.plot(df_20['Pressure'], df_20['Buckle Force'], marker='o', label='20mm', linestyle='', color=colors[0])
    plt.plot(df['Pressure'].unique(), bl_20, label = '20mm Dimeter Theoretical Collapse Force', linestyle= '-', color = colors[0], alpha = 0.8)
    df_25 = df[df['Diameter'] == 25]
    plt.plot(df_25['Pressure'], df_25['Buckle Force'], marker='^', label = '25mm', linestyle='', color=colors[0])
    plt.plot(df['Pressure'].unique(), bl_25, label = '25mm Diameter Theoretical Collapse Force', linestyle= '-', color = colors[0], alpha = 0.6)
    df_28 = df[df['Diameter'] == 28]
    plt.plot(df_28['Pressure'], df_28['Buckle Force'], marker='x', label = '28mm', linestyle='', color=colors[0])
    plt.plot(df['Pressure'].unique(), bl_28, label = '28mm Diameter Theoretical Collapse Force', linestyle= '-', color = colors[0], alpha = 0.4)
    plt.xlabel('Pressure (Bar)')
    plt.ylabel('Buckle Force (N)')
    plt.title('60A Pressure vs. Buckle Force')
    plt.legend()
    plt.grid(True)
    
    df = spring_data[(spring_data['Material'] == 70) & (spring_data['Pressure'] >= 0)]
    plt.subplot(1, 3, 2)
    df_16 = df[df['Diameter'] == 16]
    plt.plot(df_16['Pressure'], df_16['Buckle Force'], marker='s', label='16mm', linestyle='', color=colors[1])
    plt.plot(df['Pressure'].unique(), bl_16,label = '16mm Diameter Theoretical Collapse Force', linestyle= '-', color = colors[1], alpha = 1)
    df_20 = df[df['Diameter'] == 20]
    plt.plot(df_20['Pressure'], df_20['Buckle Force'], marker='o', label='20mm', linestyle='', color=colors[1])
    plt.plot(df['Pressure'].unique(), bl_20, label = '20mm Dimeter Theoretical Collapse Force', linestyle= '-', color = colors[1], alpha = 0.8)
    df_25 = df[df['Diameter'] == 25]
    plt.plot(df_25['Pressure'], df_25['Buckle Force'], marker='^', label = '25mm', linestyle='', color=colors[1])
    plt.plot(df['Pressure'].unique(), bl_25, label = '25mm Diameter Theoretical Collapse Force', linestyle= '-', color = colors[1], alpha = 0.6)
    df_28 = df[df['Diameter'] == 28]
    plt.plot(df_28['Pressure'], df_28['Buckle Force'], marker='x', label = '28mm', linestyle='', color=colors[1])
    plt.plot(df['Pressure'].unique(), bl_28, label = '28mm Diameter Theoretical Collapse Force', linestyle= '-', color = colors[1], alpha = 0.4)
    plt.xlabel('Pressure (Bar)')
    plt.ylabel('Buckle Force (N)')
    plt.title('70A Pressure vs. Buckle Force')
    plt.legend()
    plt.grid(True)

    df = spring_data[(spring_data['Material'] == 82) & (spring_data['Pressure'] >= 0)]
    plt.subplot(1, 3, 3)
    df_16 = df[df['Diameter'] == 16]
    plt.plot(df_16['Pressure'], df_16['Buckle Force'], marker='s', label='16mm', linestyle='', color=colors[2])
    plt.plot(df['Pressure'].unique(), bl_16,label = '16mm Diameter Theoretical Collapse Force', linestyle= '-', color = colors[2], alpha = 1)
    df_20 = df[df['Diameter'] == 20]
    plt.plot(df_20['Pressure'], df_20['Buckle Force'], marker='o', label='20mm', linestyle='', color=colors[2])
    plt.plot(df['Pressure'].unique(), bl_20,label = '20mm Diameter Theoretical Collapse Force', linestyle= '-', color = colors[2], alpha = 0.8)
    df_25 = df[df['Diameter'] == 25]
    plt.plot(df_25['Pressure'], df_25['Buckle Force'], marker='^', label = '25mm', linestyle='', color=colors[2])
    plt.plot(df['Pressure'].unique(), bl_25,label = '25mm Diameter Theoretical Collapse Force', linestyle= '-', color = colors[2], alpha = 0.6)
    df_28 = df[df['Diameter'] == 28]
    plt.plot(df_28['Pressure'], df_28['Buckle Force'], marker='x', label = '28mm', linestyle='', color=colors[2])
    plt.plot(df['Pressure'].unique(), bl_28,label = '28mm Diameter Theoretical Collapse Force', linestyle= '-', color = colors[2], alpha = 0.4)
    plt.xlabel('Pressure (Bar)')
    plt.ylabel('Buckle Force (N)')
    plt.title('82A Pressure vs. Buckle Force')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def optimal_data_prep(weights,spring_data,speed_data,diameter_data):
    # for all configs at p >= 0 
    # we compute our score per sample where we multiply each parameter: unload work, total vol, speed, max vac retract force  
    # we then have a plot of diameter vs material and score
    spring_data = spring_data.dropna()
    speed_data = speed_data.dropna()
    diameter_data = diameter_data.dropna()
    materials_internal = [60,70,82]
    diameters_internal = [16,20,24,28]
    pressures_internal = [0,50,75,100,125,150]
    spring_df = spring_data[(spring_data['Pressure'] >= 0)]
    columns_to_scale = ['Unload Work']
    scaler = MinMaxScaler(feature_range=(-10,10))
    spring_df[columns_to_scale] = scaler.fit_transform(spring_df[columns_to_scale])
    print(spring_df['Unload Work'])
    spring_df['ULW Score'] = spring_df['Unload Work'] * weights[0]
    spring_df_vac_peak = spring_data[spring_data['Pressure'] == -50]
    columns_to_scale = ['Peak Load Force']
    spring_df_vac_peak['Peak Load Force'] = 1 / spring_df_vac_peak['Peak Load Force']
    scaler = MinMaxScaler(feature_range=(-10,10))
    spring_df_vac_peak[columns_to_scale] = scaler.fit_transform(spring_df_vac_peak[columns_to_scale])
    spring_df_vac_peak['VRF Score'] = spring_df_vac_peak['Peak Load Force'] * weights[1]
    speed_df = speed_data[(speed_data['Pressure'] >= 0)]
    columns_to_scale = ['Angular Rate']
    scaler = MinMaxScaler(feature_range=(-10,10))
    speed_df[columns_to_scale] = scaler.fit_transform(speed_df[columns_to_scale])
    speed_df['ERT Score'] = speed_df['Angular Rate'] * weights[2]
    speed_df['Diameter'] = speed_df['Diameter'].astype(float)
    speed_df['Material'] = speed_df['Material'].astype(float)
    speed_df['Pressure'] = speed_df['Pressure'].astype(float)
    vac_vols = diameter_data.loc[diameter_data['Pressure'] == -50] 
    diameter_df = diameter_data[diameter_data['Pressure'] >= 0]
    subtracted_df = pd.merge(diameter_df, vac_vols, on=['Diameter','Material'],suffixes=('_df1', '_vac'))
    subtracted_df['Delta Vol2'] = subtracted_df['Delta Vol2_df1'] + subtracted_df['Delta Vol2_vac']
    columns_to_scale = ['Delta Vol2']
    subtracted_df['Delta Vol2'] = 1 / subtracted_df['Delta Vol2']
    scaler = MinMaxScaler(feature_range=(-10,10))
    subtracted_df[columns_to_scale] = scaler.fit_transform(subtracted_df[columns_to_scale])
    subtracted_df['Delta Vol2'] *= weights[3]
    save_diameters = []
    save_pressures = []
    save_materials = []
    save_delv2 = []
    save_peak = []
    save_work = []
    save_speed = []
    save_ulw = []
    save_vrf = []
    save_tdt = []
    save_tvc = []
    save_total = []
    for d in diameters_internal:
        for m in materials_internal:
            for pres in pressures_internal:
                val = (spring_df['Diameter'] == d) & (spring_df['Material'] == m) & (spring_df['Pressure'] == pres)
                val2 = speed_df[(speed_df['Diameter'] == d) & (speed_df['Material'] == m) & (speed_df['Pressure'] == pres)]
                val3 = subtracted_df[(subtracted_df['Diameter'] == d) & (subtracted_df['Material'] == m) & (subtracted_df['Pressure_df1'] == pres)].copy()
                val3['TVC Score'] = val3.iloc[0]['Delta Vol2'] 
                val4 = (spring_df_vac_peak['Diameter'] == d) & (spring_df_vac_peak['Material'] == m)
                #print(val3)
                #val2 = (speed_df['Diameter'] == d) & (speed_df['Material'] == m) & (speed_df['Pressure'] == pres)
                #tdt_score = val2.iloc[0]['TDT Score']
                try:
                    ulw_score = spring_df.loc[val, 'ULW Score'].iloc[0]
                    save_work.append(spring_df.loc[val,'Unload Work'].iloc[0])
                except Exception:
                    print('ulw does not exist')
                    ulw_score = np.nan
                    save_work.append(np.nan)
                try:
                    vrf_score = spring_df_vac_peak.loc[val4,'VRF Score'].iloc[0]
                    save_peak.append(spring_df_vac_peak.loc[val4,'Peak Load Force'])
                except Exception:
                    print('vrf doesnt exist')
                    vrf_score = np.nan
                    save_peak.append(np.nan)
                try:
                    tdt_score = val2.iloc[0]['ERT Score']
                    save_speed.append(val2.iloc[0]['Angular Rate'])
                except Exception:
                    print('tdt does not exist')
                    tdt_score = np.nan
                    save_speed.append(np.nan)
                try:
                    tvc_score = val3['TVC Score'].iloc[0]
                    save_delv2.append(val3['Delta Vol2'].iloc[0])
                except Exception:
                    print('tvc does not exist')
                    tvc_score = np.nan
                    save_delv2.append(np.nan)
                total_score = ulw_score + (vrf_score) + (tdt_score) + (tvc_score)
                #print(d,m,pres,total_score)
                row = [d,m,pres,ulw_score, vrf_score, tdt_score, tvc_score, total_score]
                save_diameters.append(row[0])
                save_materials.append(row[1])
                save_pressures.append(row[2])
                save_ulw.append(row[3])
                save_vrf.append(row[4])
                save_tdt.append(row[5])
                save_tvc.append(row[6])
                save_total.append(row[7])
    save = {"Diameter" : save_diameters, "Material" : save_materials, "Pressure" : save_pressures, "Unload Work" : save_work, "Peak Load Force" : save_peak, "Angular Rate" : save_speed, "Delta Vol2" : save_delv2, "ULW" : save_ulw, "VRF" : save_vrf, "ERT" : save_tdt, "TVC" : save_tvc, "Total" : save_total}
    scores_df = pd.DataFrame(save)
    return scores_df
    #print(combined_df)

def make_radar_chart(name,data,categories,color, total):
     # Number of variables we're plotting.
    num_vars = len(categories)

    # Compute angle each bar is centered on:
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # The plot is made in a circle, so we need to "complete the loop" and append the start value to the end.
    data += data[:1]
    angles += angles[:1]

    # Draw the plot
    ax = plt.subplot(1, 1, 1, polar=True)
    
    # Draw the outline of our data
    ax.fill(angles, data, color=color, alpha=0.25)
    ax.plot(angles, data, color=color, linewidth=2, label=name)
    ax.tick_params(axis='y', labelsize=10, pad=20)
    # Labels for each category
    plt.xticks(angles[:-1], categories, fontsize = 12)
    ax.tick_params(axis='x', labelsize=12, pad=10)
    #plt.text(0.95,0.5, f'{name} score: {total}')

    # Title of each subplot
    plt.title('Spider Chart Comparing Best Configuration per Material', fontsize = 14)

    return plt


# Theory Section -----
bl_16 = []
bl_20 = []
bl_25 = []
bl_28 = []
k_16_60 = []
k_20_60 = []
k_25_60 = []
k_28_60 = []
k_16_70 = []
k_20_70 = []
k_25_70 = []
k_28_70 = []
k_16_82 = []
k_20_82 = []
k_25_82 = []
k_28_82 = []

def buckle_load(p,r,L):
    return (np.pi * p * r **3) / (2*L)

def stiffness(E,I,L):
    return (E*I) / L
    #return (3*E*I) / L**3

def spring_const(E,I):
    return 3*E * I

def gent(s):
    return 1000000 * (0.0981 * (56 + 7.66 * s)) / (0.137505 * (254 - 2.54 * s)) 

def inertia(r,t):
    return np.pi*(r**3) * t

for p in pressures:
    bl_16.append(buckle_load(p,rs[0],L + 0.05))
    bl_20.append(buckle_load(p,rs[1],L + 0.05))
    bl_25.append(buckle_load(p,rs[2],L + 0.05))
    bl_28.append(buckle_load(p,rs[3],L + 0.05))

k_16_60.append(stiffness(gent(60),inertia(rs[0], 0.0012),L))
k_20_60.append(stiffness(gent(60),inertia(rs[1], 0.0012),L))
k_25_60.append(stiffness(gent(60),inertia(rs[2], 0.0012),L))
k_28_60.append(stiffness(gent(60),inertia(rs[3], 0.0012),L))

k_16_70.append(stiffness(gent(70),inertia(rs[0], 0.0012),L))
k_20_70.append(stiffness(gent(70),inertia(rs[1], 0.0012),L))
k_25_70.append(stiffness(gent(70),inertia(rs[2], 0.0012),L))
k_28_70.append(stiffness(gent(70),inertia(rs[3], 0.0012),L))

k_16_82.append(stiffness(gent(82),inertia(rs[0], 0.0012),L))
k_20_82.append(stiffness(gent(82),inertia(rs[1], 0.0012),L))
k_25_82.append(stiffness(gent(82),inertia(rs[2], 0.0012),L))
k_28_82.append(stiffness(gent(82),inertia(rs[3], 0.0012),L))

def slope(E,I,L,r_test):
    return (np.pi*E*I) / (L**2)


def spring_const_leo(k_0,p,r_0,E,t_0):
    return (k_0 * (2 * ((p*r_0) / (E * t_0)) + ((p*r_0) / (E * t_0))**2)) + k_0

# End Theory Section ----------
spring_data_avg['Unbuckle Slope'] = spring_data_avg['Unbuckle Slope'] * 57.29
es_60 = spring_data_avg[(spring_data_avg['Material'] == 60)] 
es_70 = spring_data_avg[(spring_data_avg['Material'] == 70)] 
es_82 = spring_data_avg[(spring_data_avg['Material'] == 82)] 
diameters_np = np.linspace(16,28,100)
#diameters_np = [16,20,25,28]
material_np = [60,70,82]
pressures_np = [50000,75000,100000,125000,150000]
ds = []
ms = []
stiffs = []
stiffs_l = []
ps = []
for d in diameters_np:
    for mat in material_np:
        for p in pressures_np:
            r = d / 2000
            df = spring_data_avg[(spring_data_avg['Material'] == mat) & (spring_data_avg['Pressure'] == 0) & (spring_data_avg['Diameter'] == d)]
            stiff = stiffness(gent(mat), inertia(r,(1.2 / 1000)), L)
            #stiff = spring_const(gent(mat),inertia(r, 1.2/100))
            #stiff = (np.pi*gent(mat)*inertia(r,1.2 / 100)) / (L**2)
            #stiff_l = spring_const_leo(df['Unbuckle Slope'].iloc[0],p,r,gent(mat),1.2 / 1000)
            stiff_l = spring_const_leo(stiff,p,r,gent(mat),1.2 / 1000)
            ds.append(d)
            ms.append(mat)
            ps.append(p)
            stiffs.append(stiff)
            stiffs_l.append(stiff_l)
save = {"Diameter" : ds, "Pressure" : ps, "Material" : ms, "Slope" : stiffs, "Slope2" : stiffs_l}
theory_stiffness_df = pd.DataFrame(save)
print(theory_stiffness_df)

#ts_16 = theory_stiffness_df[theory_stiffness_df['Diameter'] == 16]
#ts_20 = theory_stiffness_df[theory_stiffness_df['Diameter'] == 20]
#ts_25 = theory_stiffness_df[theory_stiffness_df['Diameter'] == 25]
#ts_28 = theory_stiffness_df[theory_stiffness_df['Diameter'] == 28]
ts_60 = theory_stiffness_df[theory_stiffness_df['Material'] == 60]
ts_70 = theory_stiffness_df[theory_stiffness_df['Material'] == 70]
ts_82 = theory_stiffness_df[theory_stiffness_df['Material'] == 82]
#plt.figure(figsize=(12, 5))
#plt.plot(ts_60['Diameter'], ts_60['Slope'], label='60')
#plt.plot(ts_70['Diameter'], ts_70['Slope'], label='70')
#plt.plot(ts_82['Diameter'], ts_82['Slope'], label='82')
#plt.plot(es_60['Diameter'], es_60['Unbuckle Slope'], linestyle='--', label='60 exp')
#plt.plot(es_70['Diameter'], es_70['Unbuckle Slope'], linestyle='--', label='70 exp')
#plt.plot(es_82['Diameter'], es_82['Unbuckle Slope'], linestyle='--', label='82 exp')
#plt.xlabel('Diameter')
#plt.ylabel('Buckle Force')
#plt.title('82A Pressure vs. Buckle Force')
#plt.legend()
#plt.grid(True)
#plt.tight_layout()
#plt.show()

dfs = [(es_60, '#1aaf6c'), (es_70, '#429bf4'), (es_82, '#d42cea')]
#dfs = [(es_60, '#1aaf6c')]

def adjust_color_brightness(color, amount=0.5):
    import matplotlib.colors as mcolors
    rgb = mcolors.to_rgb(color)
    
    # Scale the RGB values by the amount and clamp each to the range [0, 1]
    adjusted_rgb = np.clip(np.array(rgb) * amount, 0, 1)

    # Convert back to hex and return
    return mcolors.to_hex(adjusted_rgb)

# Plotting
#plt.figure(figsize=(12, 5))
#plt.subplot(1,3,1)
pressures = [50, 75, 100, 125, 150]
brightness_factors = np.linspace(0.8, 1.2, len(pressures))  # Darken or lighten the color
i = 1
for df, base_color in dfs:
    for pressure, factor in zip(pressures, brightness_factors):
        subset = df[df['Pressure'] == pressure]
        #plt.subplot(1,3,i)
        #plt.scatter(subset['Diameter'], subset['Unbuckle Slope'], color=adjust_color_brightness(base_color, factor), alpha = 0.5,label=f'{pressure} kPa')
    i += 1
#plt.subplot(1,3,1)
#plt.plot(ts_60['Diameter'], ts_60['Slope'],color='#1aaf6c', label='60A Theory')
#plt.subplot(1,3,2)
#plt.plot(ts_70['Diameter'], ts_70['Slope'],color='#429bf4', label='70A Theory')
#plt.subplot(1,3,3)
#plt.plot(ts_82['Diameter'], ts_82['Slope'],color='#d42cea', label='82A Theory')
ts_60 = ts_60[ts_60['Pressure'] == 150000]
ts_70 = ts_70[ts_70['Pressure'] == 150000]
ts_82 = ts_82[ts_82['Pressure'] == 150000]
#plt.subplot(1,3,1)
#plt.plot(ts_60['Diameter'], ts_60['Slope2'],color='#1aaf6c',alpha=0.5, label='60A at 150kPa Theory')
#plt.legend(title="Pressure")
#plt.xlabel('Diameter (mm)')
#plt.ylabel('Bending Sprint Constant (Nm/rad)')
#plt.title('Diameter vs Bending Spring Constant for Tested Pressures (60A)')
#plt.subplot(1,3,2)
#plt.plot(ts_70['Diameter'], ts_70['Slope2'],color='#429bf4',alpha=0.5, label='70A at 150kPa Theory')
#plt.legend(title="Pressure")
#plt.xlabel('Diameter (mm)')
#plt.ylabel('Bending Sprint Constant (Nm/rad)')
#plt.title('Diameter vs Bending Spring Constant for Tested Pressures (70A)')
#plt.subplot(1,3,3)
#plt.plot(ts_82['Diameter'], ts_82['Slope2'],color='#d42cea',alpha=0.5, label='82A at 150kPa Theory')
# Improve legend and plot aesthetics
#plt.legend(title="Pressure")
#plt.xlabel('Diameter (mm)')
#plt.ylabel('Bending Sprint Constant (Nm/rad)')
#plt.title('Diameter vs Bending Spring Constant for Tested Pressures (82A)')
#plt.tight_layout()
#plt.show()

# making a note of the above plot so the issue was not accounting for the moment at the end of the beam as a result of the rigid component attached to it 
# with this in mind the eq for the stiffness pertains to a tip moment which then gives us a spring constant for torsion. accounting for the 

# so the ideal output of this optimziation would be that in combination with theory you could simulate the options


#print(multi_sorted_df)
#optimal2(spring_data, speed_data, diameter_data)

# 1 Plots peak loading force at vacuum and shows us our trend between material and diameter 
# add some pictures of the cross section 

# -------- PAPER FIGURE 5 --------
#figure_1()
# -------------------------------

# 2 Plots unloading work across the full tested range and shows us the trend between diameter and pressure for each respecteve material
#figure_3_pannel_cont(spring_data,'Unload Work', '(J)')
#alts.figure_2_alt(spring_data, spring_data_avg) 


# 3 Plots the fitted slope for the two portions i think the non cont plot makes most sense here 

#figure_3_pannel_cont(spring_data, 'Buckle Slope')
#figure_3_pannel_cont(spring_data, 'Unbuckle Slope')

#labels = ['Pressure (Bar)', 'Spring Constant (Nm / rad)', 'Spring Constant Vs. Pressure']
#alts.figure_3_pannel(spring_data_avg,'Buckle Slope',labels)
#alts.figure_3_pannel(spring_data_avg,'Unbuckle Slope',labels)

#labels = ['Pressure (Bar)', 'Spring Constant (Nm / rad)', 'Buckled Intercept Vs. Pressure']
#alts.figure_3_pannel(spring_data_avg,'Buckle Intercept',labels)
#alts.figure_3_pannel(theory_stiffness_df,'Slope')
#alts.figure_3_pannel(spring_data_avg,'Buckled R2')
# 
#alts.figure_3_pannel(spring_data_avg,'Unbuckle Slope')
#diameter_data_pos = diameter_data_avg[diameter_data_avg['Pressure'] >= 0]
#alts.figure_3_pannel(diameter_data_pos, 'New Diameter')

diameter_data_avg['Percent Change'] = ((diameter_data_avg['New Diameter'] - diameter_data_avg['Diameter']) / diameter_data_avg['Diameter']) * 100
diameter_data_pos = diameter_data_avg[diameter_data_avg['Pressure'] >= 0]
#alts.figure_3_pannel(diameter_data_pos, 'Percent Change')
spring_data_hyst = spring_data_avg[spring_data_avg['Pressure'] >= 0]
#alts.figure_3_pannel(spring_data_avg, 'Hysteresis')

#alts.figure_3_pannel(spring_data_avg,'Unbuckle R2')
# here we can compare to euler beam theory for the thing and see how it compares 
# it maybe should be quite close
# this is showing us that the stiffnes is mainly determined by the material except for the cases where there is significant diameter change 

# 4 Plots the theoretical buckling forces for thin membraned materials at various pressures - here we can see the material thickness plays a role 
#figure_4b(bl_16,bl_20,bl_25,bl_28,spring_data_avg)

# 5 Plots the full time to extend vs Pressure for the various materials all on one plot
#alts.figure_5_alt(speed_data_avg)

#alts.figure_3_pannel(speed_data,'Dt1')
#alts.figure_3_pannel(speed_data,'Dt2')
#alts.figure_3_pannel(speed_data,'Dt3')
#alts.figure_3_pannel(speed_data,'Full dt')::w

#labels = ['Pressure (Bar)', 'Zeta', 'Zeta VS Pressure for Tested Diameters']
#alts.figure_3_pannel(speed_data_avg,'Zeta',labels)
speed_data_avg = speed_data_avg[~((speed_data_avg['Diameter'] == 28) & (speed_data_avg['Pressure'] == 150))]

# ------ PAPER FIGURE 7 ------
#figure_3_pannel_cont(spring_data_avg,'Unload Work', '(J)')
# -------------------------

# ------ PAPER FIGURE 8 -----
#figure_3_pannel_cont(speed_data_avg, 'Angular Rate','(Deg/s)')
# ------------------------

# 6 Plots the change in volume required to reach the desired presure
# -------- PAPER FIGURE 6 -----------
#total_vol_change(diameter_data_avg, diameter_data)
# ---------------------------------

#labels = ['Pressure (Bar)', 'Total Volume Change (mL)', 'Total Volume Change VS Pressure for tested Diameters']
#alts.figure_3_pannel(diameter_data_avg,'Delta Vol2',labels,diameter_data)

# 7 Plots the diameter measured at the given pressure
#alts.figure_3_pannel(diameter_data,'New Diameter')


categories = ['ULW', 'VRF', 'ERT', 'TVC']
spring_data_avg = spring_data_avg[spring_data_avg['Peak Load Force'] < 0.55]
weights1 = [6,5,9,10]
weights2 = [1,5,5,1]
weights_video = [2,3,4,10]
weights_video2 = [10,7,3,4]
weights_video3 = [5,5,5,5]
scores_df = optimal_data_prep(weights1, spring_data_avg, speed_data_avg, diameter_data_avg)
scores_df = scores_df.dropna()
print(scores_df)
#scaler = MinMaxScaler(feature_range=(0,10))
#columns_to_scale = categories
#scores_df[columns_to_scale] = scaler.fit_transform(scores_df[columns_to_scale])
df_60 = scores_df[scores_df['Material'] == 60]
df_70 = scores_df[scores_df['Material'] == 70]
df_82 = scores_df[scores_df['Material'] == 82]
df_60_sorted = df_60.sort_values(by='Total',ascending= False)
df_70_sorted = df_70.sort_values(by='Total',ascending= False)
df_82_sorted = df_82.sort_values(by='Total',ascending= False)
#print(df_82_sorted['Total'].iloc[0])
#print(df_70_sorted['Total'].iloc[0])
#print(df_60_sorted['Total'].iloc[0])
df_16 = scores_df[scores_df['Diameter'] == 16]
df_20 = scores_df[scores_df['Diameter'] == 20]
df_25 = scores_df[scores_df['Diameter'] == 25]
df_28 = scores_df[scores_df['Diameter'] == 28]
df_16_sorted = df_16.sort_values(by='Total', ascending = False)
df_20_sorted = df_20.sort_values(by='Total', ascending = False)
df_25_sorted = df_25.sort_values(by='Total', ascending = False)
df_28_sorted = df_28.sort_values(by='Total', ascending = False)
#print('-----')
#print(df_16_sorted['Total'].iloc[0])
#print(df_20_sorted['Total'].iloc[0])
#print(df_25_sorted['Total'].iloc[0])
#print(df_28_sorted['Total'].iloc[0])
multi_sorted_df = scores_df.sort_values(by='Total',ascending= False)
diameter_60 = df_60_sorted['Diameter'].iloc[0]
pressure_60 = df_60_sorted['Pressure'].iloc[0]
t_60 = df_60_sorted['Total'].iloc[0]
diameter_70 = df_70_sorted['Diameter'].iloc[0]
pressure_70 = df_70_sorted['Pressure'].iloc[0]
t_70 = df_70_sorted['Total'].iloc[0]
diameter_82 = df_82_sorted['Diameter'].iloc[0]
pressure_82 = df_82_sorted['Pressure'].iloc[0]
t_82 = df_82_sorted['Total'].iloc[0]
names = [f'60A, {diameter_60}mm, {pressure_60 / 100} Bar \n Score: {t_60:.2f}', f'70A, {diameter_70}mm, {pressure_70 / 100} Bar \n Score: {t_70:.2f}',
 f'82A, {diameter_82}mm, {pressure_82/ 100} Bar \n Score: {t_82:.2f}']
df_60_sorted = df_60_sorted[categories]
df_70_sorted = df_70_sorted[categories]
df_82_sorted = df_82_sorted[categories]
dataframes_mat = [df_60_sorted,df_70_sorted, df_82_sorted]
colors = ['#1aaf6c','#429bf4','#d42cea']
plt.figure(figsize=(5, 5),dpi=300)
for i, df in enumerate(dataframes_mat):
    data = df.iloc[0].tolist()
    make_radar_chart(names[i],data,categories, colors[i],0)
plt.tight_layout()
plt.legend(loc='upper right',bbox_to_anchor=(1.05,1), fontsize=10)
plt.show()