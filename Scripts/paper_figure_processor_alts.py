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



def figure_3_pannel(data_in,key,labels, data_in_raw = None):
    data = data_in.copy()
    data['Pressure'] = data['Pressure'] / 100
    df = data[data['Material'] == 60]
    plt.figure(figsize=(15, 5))
    colors =  ['#1aaf6c','#429bf4','#d42cea']
    x_err = 0.1
    # Plot Pressure vs. buckle
    plt.subplot(1, 3, 1)
    df_16 = df[df['Diameter'] == 16]
    plt.errorbar(df_16['Pressure'], df_16[key], xerr=x_err, marker='s', label='16mm', linestyle='--', color=colors[0])
    df_20 = df[df['Diameter'] == 20]
    plt.errorbar(df_20['Pressure'], df_20[key], xerr=x_err, marker='o', label='20mm', linestyle='--', color=colors[0])
    df_24 = df[df['Diameter'] == 24]
    plt.errorbar(df_24['Pressure'], df_24[key], xerr=x_err, marker='^', label = '24mm', linestyle='--', color=colors[0])
    df_28 = df[df['Diameter'] == 28]
    plt.errorbar(df_28['Pressure'], df_28[key], xerr=x_err, marker='x', label = '28mm', linestyle='--', color=colors[0])
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.title(f'60A {labels[2]}')
    plt.legend()
    plt.grid(True)
    
    df = data[data['Material'] == 70]
    plt.subplot(1, 3, 2)
    df_16 = df[df['Diameter'] == 16]
    plt.errorbar(df_16['Pressure'], df_16[key], xerr=x_err, marker='s', label='16mm', linestyle='--', color=colors[1])
    df_20 = df[df['Diameter'] == 20]
    plt.errorbar(df_20['Pressure'], df_20[key], xerr=x_err, marker='o', label='20mm', linestyle='--', color=colors[1])
    df_24 = df[df['Diameter'] == 24]
    plt.errorbar(df_24['Pressure'], df_24[key], xerr=x_err, marker='^', label = '24mm', linestyle='--', color=colors[1])
    df_28 = df[df['Diameter'] == 28]
    plt.errorbar(df_28['Pressure'], df_28[key], xerr=x_err, marker='x', label = '28mm', linestyle='--', color=colors[1])
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.title(f'70A {labels[2]}')
    plt.legend()
    plt.grid(True)

    df = data[data['Material'] == 82]
    plt.subplot(1, 3, 3)
    df_16 = df[df['Diameter'] == 16]
    plt.errorbar(df_16['Pressure'], df_16[key], xerr=x_err, marker='s', label='16mm', linestyle='--', color=colors[2])
    df_20 = df[df['Diameter'] == 20]
    plt.errorbar(df_20['Pressure'], df_20[key], xerr=x_err, marker='o', label='20mm', linestyle='--', color=colors[2])
    df_24 = df[df['Diameter'] == 24]
    plt.errorbar(df_24['Pressure'], df_24[key], xerr=x_err, marker='^', label = '24mm', linestyle='--', color=colors[2])
    df_28 = df[df['Diameter'] == 28]
    plt.errorbar(df_28['Pressure'], df_28[key], xerr=x_err, marker='x', label = '28mm', linestyle='--', color=colors[2])
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.title(f'82A {labels[2]}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def figure_2_alt(spring_data,spring_data_avg):
    df_og = spring_data[spring_data['Material'] == 60]
    df = spring_data_avg[spring_data_avg['Material'] == 60]
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    x_err = 10
    df_16 = df[df['Diameter'] == 16]
    df_og = df_og[df_og['Diameter'] == 16]
    plt.errorbar(df_16['Pressure'], df_16['Unload Work'],xerr=x_err,yerr=df_og['Unload Work'].sem(), marker='s', label='16mm', linestyle='--', color='b')
    df_20 = df[df['Diameter'] == 20]
    df_og = df_og[df_og['Diameter'] == 20]
    plt.errorbar(df_20['Pressure'], df_20['Unload Work'],xerr=x_err,yerr=df_og['Unload Work'].sem(), marker='o', label='20mm', linestyle='--', color='b')
    df_25 = df[df['Diameter'] == 25]
    df_og = df_og[df_og['Diameter'] == 25]
    plt.errorbar(df_25['Pressure'], df_25['Unload Work'],xerr=x_err,yerr=df_og['Unload Work'].sem(), marker='^', label = '25mm', linestyle='--', color='b')
    df_28 = df[df['Diameter'] == 28]
    df_og = df_og[df_og['Diameter'] == 28]
    plt.errorbar(df_28['Pressure'], df_28['Unload Work'],xerr=x_err,yerr=df_og['Unload Work'].sem(), marker='x', label = '28mm', linestyle='--', color='b')
    plt.xlabel('Pressure')
    plt.ylabel('Unload Work')
    plt.title('60A Pressure vs. Work')
    plt.legend()
    plt.grid(True)

    df_og = spring_data[spring_data['Material'] == 70]
    df = spring_data_avg[spring_data_avg['Material'] == 70]
    plt.subplot(1, 3, 2)
    df_16 = df[df['Diameter'] == 16]
    df_og = df_og[df_og['Diameter'] == 16]
    plt.errorbar(df_16['Pressure'], df_16['Unload Work'],xerr=x_err, yerr=df_og['Unload Work'].sem(), marker='s', label='16mm', linestyle='--', color='r')
    df_20 = df[df['Diameter'] == 20]
    df_og = df_og[df_og['Diameter'] == 20]
    plt.errorbar(df_20['Pressure'], df_20['Unload Work'],xerr=x_err,yerr=df_og['Unload Work'].sem(), marker='o', label='20mm', linestyle='--', color='r')
    df_25 = df[df['Diameter'] == 25]
    df_og = df_og[df_og['Diameter'] == 25]
    plt.errorbar(df_25['Pressure'], df_25['Unload Work'],xerr=x_err,yerr=df_og['Unload Work'].sem(), marker='^', label = '25mm', linestyle='--', color='r')
    df_28 = df[df['Diameter'] == 28]
    df_og = df_og[df_og['Diameter'] == 28]
    plt.errorbar(df_28['Pressure'], df_28['Unload Work'],xerr=x_err,yerr=df_og['Unload Work'].sem(), marker='x', label = '28mm', linestyle='--', color='r')
    plt.xlabel('Pressure')
    plt.ylabel('Unload Work')
    plt.title('70A Pressure vs. Work')
    plt.legend()
    plt.grid(True)


    df_og = spring_data[spring_data['Material'] == 82]
    df = spring_data_avg[spring_data_avg['Material'] == 82]
    plt.subplot(1, 3, 3)
    df_16 = df[df['Diameter'] == 16]
    df_og = df_og[df_og['Diameter'] == 16]
    plt.errorbar(df_16['Pressure'], df_16['Unload Work'],xerr=x_err,yerr=df_og['Unload Work'].sem(), marker='s', label='16mm', linestyle='--', color='g')
    df_20 = df[df['Diameter'] == 20]
    df_og = df_og[df_og['Diameter'] == 20]
    plt.errorbar(df_20['Pressure'], df_20['Unload Work'],xerr=x_err, yerr=df_og['Unload Work'].sem(), marker='o', label='20mm', linestyle='--', color='g')
    df_25 = df[df['Diameter'] == 25]
    df_og = df_og[df_og['Diameter'] == 25]
    plt.errorbar(df_25['Pressure'], df_25['Unload Work'],xerr=x_err,yerr=df_og['Unload Work'].sem(), marker='^', label = '25mm', linestyle='--', color='g')
    df_28 = df[df['Diameter'] == 28]
    df_og = df_og[df_og['Diameter'] == 28]
    plt.errorbar(df_28['Pressure'], df_28['Unload Work'],xerr=x_err,yerr=df_og['Unload Work'].sem(), marker='x', label = '28mm', linestyle='--', color='g')
    plt.xlabel('Pressure')
    plt.ylabel('Unload Work')
    plt.title('82A Pressure vs. Work')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()



def figure_3a_alt(spring_data):
    df = spring_data[spring_data['Material'] == 60]
    plt.figure(figsize=(12, 5))
    # Plot Pressure vs. buckle
    plt.subplot(1, 3, 1)
    df_16 = df[df['Diameter'] == 16]
    plt.plot(df_16['Pressure'], df_16['Buckle Slope'], marker='s', label='16mm', linestyle='--', color='b')
    df_20 = df[df['Diameter'] == 20]
    plt.plot(df_20['Pressure'], df_20['Buckle Slope'], marker='o', label='20mm', linestyle='--', color='b')
    df_25 = df[df['Diameter'] == 25]
    plt.plot(df_25['Pressure'], df_25['Buckle Slope'], marker='^', label = '25mm', linestyle='--', color='b')
    df_28 = df[df['Diameter'] == 28]
    plt.plot(df_28['Pressure'], df_28['Buckle Slope'], marker='x', label = '28mm', linestyle='--', color='b')
    plt.xlabel('Pressure')
    plt.ylabel('Buckle Slope')
    plt.title('60A Pressure vs. Buckle Spring Constant')
    plt.legend()
    plt.grid(True)
    
    df = spring_data[spring_data['Material'] == 70]
    plt.subplot(1, 3, 2)
    df_16 = df[df['Diameter'] == 16]
    plt.plot(df_16['Pressure'], df_16['Buckle Slope'], marker='s', label='16mm', linestyle='--', color='r')
    df_20 = df[df['Diameter'] == 20]
    plt.plot(df_20['Pressure'], df_20['Buckle Slope'], marker='o', label='20mm', linestyle='--', color='r')
    df_25 = df[df['Diameter'] == 25]
    plt.plot(df_25['Pressure'], df_25['Buckle Slope'], marker='^', label = '25mm', linestyle='--', color='r')
    df_28 = df[df['Diameter'] == 28]
    plt.plot(df_28['Pressure'], df_28['Buckle Slope'], marker='x', label = '28mm', linestyle='--', color='r')
    plt.xlabel('Pressure')
    plt.ylabel('Buckle Slope')
    plt.title('70A Pressure vs. Buckle Spring Constant')
    plt.legend()
    plt.grid(True)

    df = spring_data[spring_data['Material'] == 82]
    plt.subplot(1, 3, 3)
    df_16 = df[df['Diameter'] == 16]
    plt.plot(df_16['Pressure'], df_16['Buckle Slope'], marker='s', label='16mm', linestyle='--', color='g')
    df_20 = df[df['Diameter'] == 20]
    plt.plot(df_20['Pressure'], df_20['Buckle Slope'], marker='o', label='20mm', linestyle='--', color='g')
    df_25 = df[df['Diameter'] == 25]
    plt.plot(df_25['Pressure'], df_25['Buckle Slope'], marker='^', label = '25mm', linestyle='--', color='g')
    df_28 = df[df['Diameter'] == 28]
    plt.plot(df_28['Pressure'], df_28['Buckle Slope'], marker='x', label = '28mm', linestyle='--', color='g')
    plt.xlabel('Pressure')
    plt.ylabel('Buckle Slope')
    plt.title('82A Pressure vs. Buckle Spring Constant')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def figure_3b_alt(spring_data):
    df = spring_data[spring_data['Material'] == 60]
    plt.figure(figsize=(12, 5))
    # Plot Pressure vs. buckle
    plt.subplot(1, 3, 1)
    df_16 = df[df['Diameter'] == 16]
    plt.plot(df_16['Pressure'], df_16['Unbuckle Slope'], marker='s', label='16mm', linestyle='--', color='b')
    df_20 = df[df['Diameter'] == 20]
    plt.plot(df_20['Pressure'], df_20['Unbuckle Slope'], marker='o', label='20mm', linestyle='--', color='b')
    df_25 = df[df['Diameter'] == 25]
    plt.plot(df_25['Pressure'], df_25['Unbuckle Slope'], marker='^', label = '25mm', linestyle='--', color='b')
    df_28 = df[df['Diameter'] == 28]
    plt.plot(df_28['Pressure'], df_28['Unbuckle Slope'], marker='x', label = '28mm', linestyle='--', color='b')
    plt.xlabel('Pressure')
    plt.ylabel('Unbuckle Slope')
    plt.title('60A Pressure vs. Unbuckle Spring Constant')
    plt.legend()
    plt.grid(True)
    
    df = spring_data[spring_data['Material'] == 70]
    plt.subplot(1, 3, 2)
    df_16 = df[df['Diameter'] == 16]
    plt.plot(df_16['Pressure'], df_16['Unbuckle Slope'], marker='s', label='16mm', linestyle='--', color='r')
    df_20 = df[df['Diameter'] == 20]
    plt.plot(df_20['Pressure'], df_20['Unbuckle Slope'], marker='o', label='20mm', linestyle='--', color='r')
    df_25 = df[df['Diameter'] == 25]
    plt.plot(df_25['Pressure'], df_25['Unbuckle Slope'], marker='^', label = '25mm', linestyle='--', color='r')
    df_28 = df[df['Diameter'] == 28]
    plt.plot(df_28['Pressure'], df_28['Unbuckle Slope'], marker='x', label = '28mm', linestyle='--', color='r')
    plt.xlabel('Pressure')
    plt.ylabel('Unbuckle Slope')
    plt.title('70A Pressure vs. Unbuckle Spring Constant')
    plt.legend()
    plt.grid(True)

    df = spring_data[spring_data['Material'] == 82]
    plt.subplot(1, 3, 3)
    df_16 = df[df['Diameter'] == 16]
    plt.plot(df_16['Pressure'], df_16['Unbuckle Slope'], marker='s', label='16mm', linestyle='--', color='g')
    df_20 = df[df['Diameter'] == 20]
    plt.plot(df_20['Pressure'], df_20['Unbuckle Slope'], marker='o', label='20mm', linestyle='--', color='g')
    df_25 = df[df['Diameter'] == 25]
    plt.plot(df_25['Pressure'], df_25['Unbuckle Slope'], marker='^', label = '25mm', linestyle='--', color='g')
    df_28 = df[df['Diameter'] == 28]
    plt.plot(df_28['Pressure'], df_28['Unbuckle Slope'], marker='x', label = '28mm', linestyle='--', color='g')
    plt.xlabel('Pressure')
    plt.ylabel('Unbuckle Slope')
    plt.title('82A Pressure vs. Unbuckle Spring Constant')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def figure_5_alt(speed_data_in):
    speed_data = speed_data_in.copy()
    speed_data['Pressure'] = speed_data['Pressure'] / 100
    df = speed_data[speed_data['Material'] == 60]
    plt.figure(figsize=(12, 5))
    # Plot Pressure vs. buckle
    colors =  ['#1aaf6c','#429bf4','#d42cea']
    plt.subplot(1, 1, 1)
    df_16 = df[df['Diameter'] == 16]
    plt.plot(df_16['Pressure'], df_16['Full dt'], marker='s', label='16mm, 60A', linestyle='--', color=colors[0], alpha = 0.6)
    df_20 = df[df['Diameter'] == 20]
    plt.plot(df_20['Pressure'], df_20['Full dt'], marker='o', label='20mm, 60A', linestyle='--', color=colors[0], alpha = 0.6)
    df_25 = df[df['Diameter'] == 25]
    plt.plot(df_25['Pressure'], df_25['Full dt'], marker='^', label = '25mm, 60A', linestyle='--', color=colors[0], alpha = 0.6)
    df_28 = df[df['Diameter'] == 28]
    plt.plot(df_28['Pressure'], df_28['Full dt'], marker='x', label = '28mm, 60A', linestyle='--', color=colors[0], alpha = 0.6)
    plt.xlabel('Pressure(Bar)')
    plt.ylabel('Time (s)')
    plt.title('60A Pressure vs. Total Time to travel 90 Deg')
    plt.legend()
    plt.grid(True)
    
    df = speed_data[speed_data['Material'] == 70]
    plt.subplot(1, 1, 1)
    df_16 = df[df['Diameter'] == 16]
    plt.plot(df_16['Pressure'], df_16['Full dt'], marker='s', label='16mm, 70A', linestyle='--', color=colors[1], alpha = 0.6)
    df_20 = df[df['Diameter'] == 20]
    plt.plot(df_20['Pressure'], df_20['Full dt'], marker='o', label='20mm, 70A', linestyle='--', color=colors[1], alpha = 0.6)
    df_25 = df[df['Diameter'] == 25]
    plt.plot(df_25['Pressure'], df_25['Full dt'], marker='^', label = '25mm, 70A', linestyle='--', color=colors[1], alpha = 0.6)
    df_28 = df[df['Diameter'] == 28]
    plt.plot(df_28['Pressure'], df_28['Full dt'], marker='x', label = '28mm, 70A', linestyle='--', color=colors[1], alpha = 0.6)
    plt.xlabel('Pressure (Bar)')
    plt.ylabel('Full dt (s)')
    plt.title('70A Pressure vs. Total time')
    plt.legend()
    plt.grid(True)

    df = speed_data[speed_data['Material'] == 82]
    plt.subplot(1, 1, 1)
    df_16 = df[df['Diameter'] == 16]
    plt.plot(df_16['Pressure'], df_16['Full dt'], marker='s', label='16mm, 82A', linestyle='--', color=colors[2], alpha = 0.6)
    df_20 = df[df['Diameter'] == 20]
    plt.plot(df_20['Pressure'], df_20['Full dt'], marker='o', label='20mm, 82A', linestyle='--', color=colors[2], alpha = 0.6)
    df_25 = df[df['Diameter'] == 25]
    plt.plot(df_25['Pressure'], df_25['Full dt'], marker='^', label = '25mm, 82A', linestyle='--', color=colors[2], alpha = 0.6)
    df_28 = df[df['Diameter'] == 28]
    plt.plot(df_28['Pressure'], df_28['Full dt'], marker='x', label = '28mm, 82A', linestyle='--', color=colors[2], alpha = 0.6)
    plt.xlabel('Pressure (Bar)')
    plt.ylabel('Full dt (s)')
    plt.title('Pressure vs. Total Time for a 90 Deg Arc')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()