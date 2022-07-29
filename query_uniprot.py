#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 16:47:49 2022

@author: xies
"""

from urllib.request import urlopen
import requests
from Bio import SeqIO
from io import StringIO


def query_uniprot_highest_hit(query_string):
    '''
    Web query of UniProt database. Returns the highest (first) hit.
    
    INPUT: query_string = string that you would type into UniProt searchbar
                e.g. 'ACT1 Organism: Mus musculus'
    OUTPUT: records
    
    '''
    
    rest_url = 'https://rest.uniprot.org/uniprotkb/search'
    
    query = {'query': query_string ,'format':'xml'}
    entries = requests.get(rest_url, query)

    recs = [rec for rec in SeqIO.parse(StringIO(entries.text),'uniprot-xml')]
    
    return recs[0]


query_string = 'ACT1 organism:human'
foo = query_uniprot_highest_hit(query_string)

