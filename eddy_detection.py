'''

  Software for the tracking of eddies in
  OFAM model output following Chelton et
  al., Progress in Oceanography, 2011.

'''

# Load required modules

import numpy as np
from matplotlib import pyplot as plt
import eddy_functions as eddy

# Load parameters

from params import *

# Load latitude and longitude vectors and restrict to domain of interest

lon, lat = eddy.load_lonlat(run)
lon, lat, i1, i2, j1, j2 = eddy.restrict_lonlat(lon, lat, lon1, lon2, lat1, lat2)

# Loop over time

lon_eddies_a = []
lat_eddies_a = []
amp_eddies_a = []
area_eddies_a = []
scale_eddies_a = []
lon_eddies_c = []
lat_eddies_c = []
amp_eddies_c = []
area_eddies_c = []
scale_eddies_c = []

for tt in range(T):

    print tt, T

# Load map of sea surface height (SSH)
 
    eta, eta_miss = eddy.load_eta(run, tt, i1, i2, j1, j2)
    eta = eddy.remove_missing(eta, missing=eta_miss, replacement=np.nan)
 
# Spatially filter SSH field
 
    eta_filt = eddy.spatial_filter(eta, lon, lat, res, cut_lon, cut_lat)
 
# Detect lon and lat coordinates of eddies

    lon_eddies, lat_eddies, amp, area, scale = eddy.detect_eddies(eta_filt, lon, lat, ssh_crits, res, Npix_min, Npix_max, amp_thresh, d_thresh, cyc='anticyclonic')
    lon_eddies_a.append(lon_eddies)
    lat_eddies_a.append(lat_eddies)
    amp_eddies_a.append(amp)
    area_eddies_a.append(area)
    scale_eddies_a.append(scale)

    lon_eddies, lat_eddies, amp, area, scale = eddy.detect_eddies(eta_filt, lon, lat, ssh_crits, res, Npix_min, Npix_max, amp_thresh, d_thresh, cyc='cyclonic')
    lon_eddies_c.append(lon_eddies)
    lat_eddies_c.append(lat_eddies)
    amp_eddies_c.append(amp)
    area_eddies_c.append(area)
    scale_eddies_c.append(scale)
 
# Plot map of filtered SSH field
 
    #plt.clf()
    #plt.contourf(lon, lat, eta_filt, levels=np.arange(-2.5,2.5,0.05))
    #plt.plot(lon_eddies_a[tt], lat_eddies_a[tt], 'k^')
    #plt.plot(lon_eddies_c[tt], lat_eddies_c[tt], 'kv')
    #plt.clim(-0.5,0.5)
    #plt.title('day: ' + str(tt))
    #plt.savefig(data_dir+'eta_filt_' + str(tt).zfill(4) + '.png', bbox_inches=0)

# Combine eddy information from all days into a list

eddies = eddy.eddies_list(lon_eddies_a, lat_eddies_a, amp_eddies_a, area_eddies_a, scale_eddies_a, lon_eddies_c, lat_eddies_c, amp_eddies_c, area_eddies_c, scale_eddies_c)

np.savez(data_dir+'eddy_det_'+run, eddies=eddies)
