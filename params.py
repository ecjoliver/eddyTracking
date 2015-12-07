import numpy as np
#import eddy_functions as eddy

def mkdir(p):
    """make directory of path that is passed"""
    import os
    try:
       os.makedirs(p)
       print "output folder: "+p+ " does not exist, we will make one."
    except OSError as exc: # Python >2.5
       import errno
       if exc.errno == errno.EEXIST and os.path.isdir(p):
          pass
       else: raise

# Eddy detection

#cb
#working folder
data_dir = '/srv/ccrc/data42/z3457920/20151012_eac_sep_dynamics/analysis/eddy_tracking/'
plot_dir = './'
plot_dir = data_dir + 'plots/'
mkdir(data_dir)
mkdir(plot_dir)

lon1 = 140 #was 90
lon2 = 180
lat1 = -55
lat2 = 0

NAME = 'cb_NEMO' # Which dataset/model run for which to detect eddies (AVISO, CTRL or A1B)

if NAME == 'CTRL':
    run = NAME
    T = 9*365 # Number of time steps to loop over
    res = 0.1 # horizontal resolution of SSH field [degrees]
    dt = 1. # Sample rate of detected eddies [days]
elif NAME == 'A1B':
    run = NAME
    T = 9*365 # Number of time steps to loop over
    res = 0.1 # horizontal resolution of SSH field [degrees]
    dt = 1. # Sample rate of detected eddies [days]
elif NAME == 'AVISO':
    run = NAME
    T = 876 # Number of time steps to loop over
    res = 0.25 # horizontal resolution of SSH field [degrees]
    dt = 7. # Sample rate of detected eddies [days]
elif NAME == 'AVISOd':
    run = NAME
    T = 7967 # Number of time steps to loop over
    res = 0.25 # horizontal resolution of SSH field [degrees]
    dt = 1. # Sample rate of detected eddies [days]
elif NAME == 'cb_AVISO':
    run = NAME
    T = 4 # Number of time steps to loop over
    res = 0.25 # horizontal resolution of SSH field [degrees]
    dt = 1. # Sample rate of detected eddies [days]
    pathroot='/srv/ccrc/data42/z3457920/RawData/AVISO/RawData/dt_global_allsat_madt/ftp.aviso.altimetry.fr/global/delayed-time/grids/madt/all-sat-merged/h/1993/'
elif NAME == 'cb_NEMO':
    run = NAME
    #T = 365 # Number of time steps to loop over
    T = 5 # Number of time steps to loop over
    res = 0.25 # horizontal resolution of SSH field [degrees]
    dt = 1. # Sample rate of detected eddies [days]
    pathroot='/srv/ccrc/data42/z3457920/20151012_eac_sep_dynamics/nemo_cordex24_ERAI01/'

cut_lon = 20. # cutoff wavelenth in longitudinal direction (for filtering) [degrees]
cut_lat = 10. # cutoff wavelenth in latitudinal direction (for filtering) [degrees]

res_aviso = 0.25 # horizontal resolution of Aviso SSH fields [degrees]

ssh_crit_max = 1.
dssh_crit = 0.01
ssh_crits = np.arange(-ssh_crit_max, ssh_crit_max+dssh_crit, dssh_crit)
ssh_crits = np.flipud(ssh_crits)

area_correction = res_aviso**2 / res**2 # correction for different resoluttions of AVISO and OFAM
Npix_min = np.floor(8*area_correction) # min number of eddy pixels
Npix_max = np.floor(1000*area_correction) # max number of eddy pixels

amp_thresh = 0.01 # minimum eddy amplitude [m]
d_thresh = 400. # max linear dimension of eddy [km] ; Only valid outside Tropics (see Chelton et al. (2011), pp. 207)

dt_aviso = 7. # Sample rate used in Chelton et al. (2011) [days]
dE_aviso = 150. # Length of search ellipse to East of eddy used in Chelton et al. (2011) [km]

#This is only used in eddy_tracking so, has been called explictly there. This removes the circular dependency so we can import params into eddy_functions!
#rossrad = eddy.load_rossrad() # Atlas of Rossby radius of deformation and first baroclinic wave speed (Chelton et al. 1998)

eddy_scale_min = 0.25 # min ratio of amplitude of new and old eddies
eddy_scale_max = 2.5 # max ratio of amplidude of new and old eddies

dt_save = 100 # Step increments at which to save data while tracking eddies
