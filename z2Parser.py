#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 12:09:28 2022

@author: xies
"""

#%%

def parse_Z2(filename):
        
    import pandas as pd
    from glob import glob
    from os import path

    # Parse a Z2 file (histogram) into a dataframe
    # Iterates through each line to find the [ demarcater and grab all numbers that comes after
    # until the next [ mark. Will only extract Bindiam or Binheight
    # @todo Need to modify to include fL measurement but that's easy
    
    f = open(filename).readlines()
    
    # start_parsing = False
    parsing = False
    df = pd.DataFrame()
    
    for line in f:
    
        #Check if we should start parsing
        if line.startswith('['):
            
            # If parsing already started, then need to save data and stop parsing
            if parsing == True:
                # save data into DF
                df[diam_or_height] = data
                # stop future parsing
                parsing = False
                diam_or_neight = 'None'
                
            
            # If parsing has not started, then check which variable to parse into
            if parsing == False and line.startswith('[#Bindiam]'):
                parsing = True
                diam_or_height = 'Diam'
                data = []
                
            elif parsing == False and line.startswith('[#Binheight]'):
                parsing = True
                diam_or_height = 'Count'
                data = []
            # elif --- here for fL if needed
            
        # Make sure not to save the lines that start with '['
        elif parsing:
            
            data.append(float(line.rstrip()))
        
        
    return df


# author: wjmallard

import pandas as pd
import numpy as np
import re

MAX_GAIN = 100

def parse_M3_file(filename):
    '''
    Parse a Beckman Coulter Multisizer 3 data file.
    '''
    data = {}
    section = None

    p_section = re.compile('^\[(.*)\]')
    p_field = re.compile('^(.*)=(.*)')

    with open(filename) as fid:
        for line in fid:

            m_section = p_section.match(line)
            m_field = p_field.match(line)

            if m_section:
                section = m_section.groups()[0]
                data[section] = {}
                continue
            elif m_field:
                k, v = m_field.groups()
                data[section][k] = v.strip()
            else:
                if 'Values' not in data[section]:
                    data[section]['Values'] = []
                data[section]['Values'].append(line.strip())

    return data

def decode_pulse(pulse_array):
    return [int(n, 16) for n in pulse_array.split(',')]

def extract_pulses(M3_data):
    '''
    Pulse hex format:
    - MaxHeight
    - MidHeight
    - Width
    - Area
    - Gain
    '''
    Pulses = M3_data['#Pulses5hex']['Values']
    return np.array([decode_pulse(p) for p in Pulses])

def filter_outliers(Pulses, threshold):

    df = pd.DataFrame(data=Pulses,
                      columns=['MaxHeight',
                               'MidHeight',
                               'Width',
                               'Area',
                               'Gain'])

    df = df[(df.MaxHeight < np.percentile(df.MaxHeight, q=threshold)) &
            (df.MidHeight < np.percentile(df.MidHeight, q=threshold)) &
            (df.Width < np.percentile(df.Width, q=threshold)) &
            (df.Area < np.percentile(df.Area, q=threshold)) &
            (df.Gain <= MAX_GAIN)]

    return df.values

def extract_height_width(filename, threshold=98):

    M3_data = parse_M3_file(filename)

    Instrument = M3_data['instrument']

    Kd = float(Instrument['Kd'])
    MaxHtCorr = int(Instrument['MaxHtCorr'])
    CountsPerVolt = 838870  # 1/V
    Current = int(Instrument['Current']) / 1000  # mA
    Resistance = 25 * int(Instrument['Gain'])  # kohm

    Pulses = extract_pulses(M3_data)
    Pulses = filter_outliers(Pulses, threshold)

    Height = Pulses[:, 0] + MaxHtCorr
    Width = Pulses[:,2]
    gain= Pulses[:,4]
    Diameter = Kd * (Height / (CountsPerVolt * Current * Resistance)) ** (1/3)

    return Diameter,Height, Width,gain

def extract_volumes(filename, threshold=98):

    Diameter = extract_diameters(filename, threshold)
    Volume = np.pi / 6 * Diameter ** 3

    return Volume

def extract_histogram(filename):

    M3_data = parse_M3_file(filename)

    BinDiameterIter = (float(x) for x in M3_data['#Bindiam']['Values'])
    BinHeightIter = (float(x) for x in M3_data['#Binheight']['Values'])

    BinDiameter = np.fromiter(BinDiameterIter, dtype=float)
    BinHeight = np.fromiter(BinHeightIter, dtype=float)

    return BinDiameter, BinHeight

