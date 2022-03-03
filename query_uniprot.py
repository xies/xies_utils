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

# Query UniProt for localization
uniprot_urlbase = 'https://www.uniprot.org/uniprot/'

        
def query_uniprot_highest_hit(query_string):
    '''
    Web query of UniProt database. Returns the highest (first) hit.
    
    INPUT: query_string = string that you would type into UniProt searchbar
                e.g. 'ACT1 Organism: Mus musculus'
    OUTPUT: records
    
    '''
    query = {"query": query_string ,'format':'fasta','sort':'score'}
    entries = requests.get(uniprot_urlbase, query)

    recs = [rec for rec in SeqIO.parse(StringIO(entries.text),'fasta')]
    rec = recs[0]
    
    upID = rec.id.split('|')[1]

    # Parse XML using Biopython
    handle = urlopen(uniprot_urlbase + upID + '.xml')

    record = SeqIO.read(handle, "uniprot-xml")
    
    return record
