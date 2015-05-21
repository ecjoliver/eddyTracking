'''

  Software for the tracking of eddies in
  OFAM model output following Chelton et
  al., Progress in Oceanography, 2011.

'''

# Load required modules

import numpy as np
import eddy_functions as eddy

# Load parameters

from params import *

# Automated eddy tracking

data = np.load(data_dir+'eddy_det_'+run+'.npz')
det_eddies = data['eddies'] # len(eddies) = number of time steps

# Initialize eddies discovered at first time step

eddies = eddy.eddies_init(det_eddies)

# Stitch eddy tracks together at future time steps

for tt in range(1, T):

    print tt, T

    # Track eddies from time step tt-1 to tt and update corresponding tracks and/or create new eddies

    eddies = eddy.track_eddies(eddies, det_eddies, tt, dt, dt_aviso, dE_aviso, rossrad, eddy_scale_min, eddy_scale_max)

    # Save data incrementally

    if( np.mod(tt, dt_save)==0 ):

        np.savez(data_dir+'eddy_track_'+run, eddies=eddies)

# Add keys for eddy age and flag if eddy was still in existence at end of run

for ed in range(len(eddies)):

    eddies[ed]['age'] = len(eddies[ed]['lon'])

np.savez(data_dir+'eddy_track_'+run, eddies=eddies)
