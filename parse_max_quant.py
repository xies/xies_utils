#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 13:18:45 2022

@author: xies
"""

import numpy as np
import pandas as pd

def filter_proteinGroup_rev_con(df):
    # Filter out Majority Protein ID that starts with REV (reverse) or CON (contaminant)
    
    
    first_id_field = df['Majority protein IDs'].str.split('_',expand=True)[0]
    I = ( first_id_field != 'CON' ) & (first_id_field != 'REV')

    return df[I].copy()

def normalize_proteinGroup(df):
    
    df['Normalized 1'] = df['Reporter intensity 1']/df['Reporter intensity 1'].sum()
    df['Normalized 2'] = df['Reporter intensity 2']/df['Reporter intensity 2'].sum()
    df['Normalized 3'] = df['Reporter intensity 3']/df['Reporter intensity 3'].sum()
    df['Normalized 4'] = df['Reporter intensity 4']/df['Reporter intensity 4'].sum()
    df['Normalized 5'] = df['Reporter intensity 5']/df['Reporter intensity 5'].sum()
    df['Normalized 6'] = df['Reporter intensity 6']/df['Reporter intensity 6'].sum()
    df['Normalized 7'] = df['Reporter intensity 7']/df['Reporter intensity 7'].sum()
    df['Normalized 8'] = df['Reporter intensity 8']/df['Reporter intensity 8'].sum()
    df['Normalized 9'] = df['Reporter intensity 9']/df['Reporter intensity 9'].sum()
    df['Normalized 10'] = df['Reporter intensity 10']/df['Reporter intensity 10'].sum()
    
    return df


