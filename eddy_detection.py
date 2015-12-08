'''

  Software for the tracking of eddies in
  OFAM model output following Chelton et
  al., Progress in Oceanography, 2011.

'''

# Load required modules

import numpy as np

import matplotlib
# Turn the followin on if you are running on storm sometimes - Forces matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')


from matplotlib import pyplot as plt
import eddy_functions as eddy

# Load parameters

from params import *

# Load latitude and longitude vectors and restrict to domain of interest

lon, lat = eddy.load_lonlat(run)
##chris' dodgy hack for not having the eric find_nearest function...
#i1=0
#i2=2000
#j1=0
#j2=2000
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

print 'eddy detection started'
print "number of time steps to loop over: ",T
for tt in range(T):
    print "timestep: ",tt+1,". out of: ", T

    # Load map of sea surface height (SSH)
 
    eta, eta_miss = eddy.load_eta(run, tt, i1, i2, j1, j2)
    eta = eddy.remove_missing(eta, missing=eta_miss, replacement=np.nan)
    #eddy.quick_plot(eta,findrange=True)
    # 
    ## Spatially filter SSH field
    # 
    eta_filt = eddy.spatial_filter(eta, lon, lat, res, cut_lon, cut_lat)
    #eddy.quick_plot(eta_filt,findrange=True)
    # 
    ## Detect lon and lat coordinates of eddies
    #
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

    eddies_a=(lon_eddies_a[tt],lat_eddies_a[tt])
    eddies_c=(lon_eddies_c[tt],lat_eddies_c[tt])
    eddy.detection_plot(tt,lon,lat,eta,eta_filt,eddies_a,eddies_c,'rawtoo',plot_dir,findrange=False)


# Combine eddy information from all days into a list

eddies = eddy.eddies_list(lon_eddies_a, lat_eddies_a, amp_eddies_a, area_eddies_a, scale_eddies_a, lon_eddies_c, lat_eddies_c, amp_eddies_c, area_eddies_c, scale_eddies_c)

np.savez(data_dir+'eddy_det_'+run, eddies=eddies)
