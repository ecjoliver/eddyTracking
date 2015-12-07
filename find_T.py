#   Author: Christopher Bull. 
#   Affiliation: Climate Change Research Centre and ARC Centre of Excellence for Climate System Science.
#                Level 4, Mathews Building
#                University of New South Wales
#                Sydney, NSW, Australia, 2052
#   Contact: z3457920@student.unsw.edu.au
#   www:     christopherbull.com.au
#   Date created: Mon, 07 Dec 2015 18:23:21
#   Machine created on: ccrc165
#

"""
Short script to find the number of time steps for the 'active' NEMO experiment (will return the one called NAME in params).
"""
from eddy_functions import raw_nemo_globber_specifytpe
import pandas as pd
import params

if __name__ == "__main__": 
    #put useful code here!
    print 'T=',len(raw_nemo_globber_specifytpe(params.pathroot,return_dates=True))
