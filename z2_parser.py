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
    
