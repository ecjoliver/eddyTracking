'''

    A set of functions to accompany the eddy tracking software

'''

import numpy as np
import scipy as sp
import numpy.linalg as linalg
import scipy.signal as signal
import scipy.ndimage as ndimage
import scipy.interpolate as interpolate
import glob
from Scientific.IO import NetCDF
from itertools import repeat
from ecoliver import find_nearest, nanmean


def load_lonlat(run, disk='erebor'):
    '''
    Loads latitude and longitude vectors from OFAM runs or AVISO
    Assumes root is /mnt/erebor unless cerberus is specified.
    '''

    #    
    # OFAM
    #

    if run=='CTRL' or run=='A1B':

        if disk=='erebor':
            pathroot = '/mnt/erebor/data/OFAM/'
        elif disk=='cerberus':
            pathroot = '/media/cerberus/data/OFAM/'

        filename = 'ctrl_nov10/daily/ocean_tse_sfc_2003_01.nc'

        fileobj = NetCDF.NetCDFFile(pathroot+filename, mode='r')
        lon = fileobj.variables['xt_ocean'].getValue()
        lat = fileobj.variables['yt_ocean'].getValue()
        fileobj.close()

    #
    # AVISO
    #

    elif run=='AVISO':

        pathroot = '/mnt/devil/data/sla/orig_qd/'

        # Find week's map
        file_header = 'dt_ref_global_merged_msla_h_qd_'
        file_list = glob.glob(pathroot + file_header + '*.nc')
        file = file_list[0]

        # load week's map
        fileobj = NetCDF.NetCDFFile(file, mode='r')
        lon = fileobj.variables['NbLongitudes'].getValue()
        lat = fileobj.variables['NbLatitudes'].getValue()
        fileobj.close()

    elif run=='AVISOd':

        pathroot = '/media/DataOne/data/sla/aviso/dt_global_twosat_msla_h/'

        # Find week's map
        file_header = 'dt_global_twosat_msla_h_'
        file_list = glob.glob(pathroot + file_header + '*.nc')
        file = file_list[0]

        # load week's map
        fileobj = NetCDF.NetCDFFile(file, mode='r')
        lon = fileobj.variables['lon'].getValue()
        lat = fileobj.variables['lat'].getValue()
        fileobj.close()

    return lon, lat


def restrict_lonlat(lon, lat, lon1, lon2, lat1, lat2):
    '''
    Restricts latitude and longitude vectors given
    input limits.
    '''

    tmp, i1 = find_nearest(lon, lon1)
    tmp, i2 = find_nearest(lon, lon2)
    tmp, j1 = find_nearest(lat, lat1)
    tmp, j2 = find_nearest(lat, lat2)

    lon = lon[i1:i2+1]
    lat = lat[j1:j2+1]

    return lon, lat, i1, i2, j1, j2


def load_eta(run, tt, i1, i2, j1, j2, disk='erebor'):
    '''
    Loads sea surface height field from OFAM runs or Aviso
    Assumes root is /mnt/erebor unless cerberus is specified.
     run = 'CTRL' or 'A1B' or 'AVISO'
    '''

    #    
    # OFAM
    #

    if run=='CTRL' or run=='A1B':

        if disk=='erebor':
            pathroot = '/mnt/erebor/data/OFAM/'
        elif disk=='cerberus':
            pathroot = '/media/cerberus/data/OFAM/'

        if run=='CTRL':
            runpath = 'ctrl_nov10/daily/'
            year0 = 1998
        elif run=='A1B':
            runpath = 'A1b_2060_oct10/daily/'
            year0 = 2065

        dpy = 365
        dpm = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        months = np.array([]).astype(int)
        for i in range(len(dpm)):
            months = np.append(months, (i+1)*np.ones(dpm[i]).astype(int))

        # which year
        year = year0 + tt/dpy

        # which month
        doy = np.mod(tt, dpy)
        month = months[doy]

        # which day
        day = doy - sum(dpm[0:month-1])

        # load day's map
        fileobj = NetCDF.NetCDFFile(pathroot + runpath + 'ocean_tse_sfc_' + str(year) + '_' +str(month).zfill(2) + '.nc', mode='r')
        eta = fileobj.variables['eta_t'].getValue()[day, j1:j2+1, i1:i2+1]
        eta_miss = fileobj.variables['eta_t']._FillValue
        fileobj.close()

    #
    # AVISO
    #

    elif run=='AVISO':

        pathroot = '/mnt/devil/data/sla/orig_qd/'

        # Find week's map
        file_header = 'dt_ref_global_merged_msla_h_qd_'
        file_list = glob.glob(pathroot + file_header + '*.nc')
        file = file_list[tt]

        # load week's map
        fileobj = NetCDF.NetCDFFile(file, mode='r')
        eta = fileobj.variables['Grid_0001'].getValue()[i1:i2+1, j1:j2+1].T / 100
        eta_miss = fileobj.variables['Grid_0001']._FillValue / 100
        fileobj.close()

    elif run=='AVISOd':

        pathroot = '/media/DataOne/data/sla/aviso/dt_global_twosat_msla_h/'

        # Find week's map
        file_header = 'dt_global_twosat_msla_h_'
        file_list = glob.glob(pathroot + file_header + '*.nc')
        file = file_list[tt]

        # load week's map
        fileobj = NetCDF.NetCDFFile(file, mode='r')
        eta = fileobj.variables['sla'].getValue()[0, j1:j2+1, i1:i2+1] * fileobj.variables['sla'].scale_factor
        eta_miss = fileobj.variables['sla']._FillValue * fileobj.variables['sla'].scale_factor
        fileobj.close()

    return eta, eta_miss[0]


def remove_missing(field, missing, replacement):
    '''
    Replaces all instances of 'missing' in 'field' with 'replacement'
    '''

    field[field==missing] = replacement

    return field


def interp_nans(data, indices):
    '''
    Linearly interpolates over missing values (np.nan's) in data
    Data is defined at locations in vector indices.
    '''

    not_nan = np.logical_not(np.isnan(data))

    return np.interp(indices, indices[not_nan], data[not_nan])


def match_missing(data1, data2):
    '''
    Make all locations that are missing in data2 also missing in data1
    Missing values are assumed to be np.nan.
    '''

    data1[np.isnan(data2)] = np.nan
    return data1


def spatial_filter(field, lon, lat, res, cut_lon, cut_lat):
    '''
    Performs a spatial filter, removing all features with
    wavelenth scales larger than cut_lon in longitude and
    cut_lat in latitude from field (defined in grid given
    by lon and lat).  Field has spatial resolution of res
    and land identified by np.nan's
    '''

    field_filt = np.zeros(field.shape)

    # see Chelton et al, Prog. Ocean., 2011 for explanation of factor of 1/5
    sig_lon = (cut_lon/5.) / res
    sig_lat = (cut_lat/5.) / res

    land = np.isnan(field)
    field[land] = nanmean(field)

    field_filt = field - ndimage.gaussian_filter(field, [sig_lat, sig_lon])

    field_filt[land] = np.nan

    return field_filt


def distance_matrix(lons,lats):
    '''Calculates the distances (in km) between any two cities based on the formulas
    c = sin(lati1)*sin(lati2)+cos(longi1-longi2)*cos(lati1)*cos(lati2)
    d = EARTH_RADIUS*Arccos(c)
    where EARTH_RADIUS is in km and the angles are in radians.
    Source: http://mathforum.org/library/drmath/view/54680.html
    This function returns the matrix.'''

    EARTH_RADIUS = 6378.1
    X = len(lons)
    Y = len(lats)
    assert X == Y, 'lons and lats must have same number of elements'

    d = np.zeros((X,X))

    #Populate the matrix.
    for i2 in range(len(lons)):
        lati2 = lats[i2]
        loni2 = lons[i2]
        c = np.sin(np.radians(lats)) * np.sin(np.radians(lati2)) + \
            np.cos(np.radians(lons-loni2)) * \
            np.cos(np.radians(lats)) * np.cos(np.radians(lati2))
        d[c<1,i2] = EARTH_RADIUS * np.arccos(c[c<1])

    return d


def detect_eddies(field, lon, lat, ssh_crits, res, Npix_min, Npix_max, amp_thresh, d_thresh, cyc='anticyclonic'):
    '''
    Detect eddies present in field which satisfy the criteria
    outlined in Chelton et al., Prog. ocean., 2011, App. B.2.

    Field is a 2D array specified on grid defined by lat and lon.

    ssh_crits is an array of ssh levels over which to perform
    eddy detection loop

    res is resolutin in degrees of field

    Npix_min, Npix_max, amp_thresh, d_thresh specify the constants
    used by the eddy detection algorithm (see Chelton paper for
    more details)

    cyc = 'cyclonic' or 'anticyclonic' [default] specifies type of
    eddies to be detected

    Function outputs lon, lat coordinates of detected eddies
    '''

    len_deg_lat = 111.325 # length of 1 degree of latitude [km]

    llon, llat = np.meshgrid(lon, lat)

    lon_eddies = np.array([])
    lat_eddies = np.array([])
    amp_eddies = np.array([])
    area_eddies = np.array([])
    scale_eddies = np.array([])

    # ssh_crits increasing for 'cyclonic', decreasing for 'anticyclonic'
    ssh_crits.sort()
    if cyc == 'anticyclonic':
        ssh_crits = np.flipud(ssh_crits)

    # loop over ssh_crits and remove interior pixels of detected eddies from subsequent loop steps
    for ssh_crit in ssh_crits:
 
    # 1. Find all regions with eta greater (less than) than ssh_crit for anticyclonic (cyclonic) eddies (Chelton et al. 2011, App. B.2, criterion 1)
        if cyc == 'anticyclonic':
            regions, nregions = ndimage.label( (field>ssh_crit).astype(int) )
        elif cyc == 'cyclonic':
            regions, nregions = ndimage.label( (field<ssh_crit).astype(int) )

        for iregion in range(nregions):
 
    # 2. Calculate number of pixels comprising detected region, reject if not within [Npix_min, Npix_max]
            region = (regions==iregion+1).astype(int)
            region_Npix = region.sum()
            eddy_area_within_limits = (region_Npix < Npix_max) * (region_Npix > Npix_min)
 
    # 3. Detect presence of local maximum (minimum) for anticylonic (cyclonic) eddies, reject if non-existent
            interior = ndimage.binary_erosion(region)
            exterior = region.astype(bool) - interior
            if interior.sum() == 0:
                continue
            if cyc == 'anticyclonic':
                has_internal_ext = field[interior].max() > field[exterior].max()
            elif cyc == 'cyclonic':
                has_internal_ext = field[interior].min() < field[exterior].min()
 
    # 4. Find amplitude of region, reject if < amp_thresh
            if cyc == 'anticyclonic':
                amp = field[interior].max() - field[exterior].mean()
            elif cyc == 'cyclonic':
                amp = field[exterior].mean() - field[interior].min()
            is_tall_eddy = amp >= amp_thresh
 
    # 5. Find maximum linear dimension of region, reject if < d_thresh
            if np.logical_not( eddy_area_within_limits * has_internal_ext * is_tall_eddy):
                continue
 
            lon_ext = llon[exterior]
            lat_ext = llat[exterior]
            d = distance_matrix(lon_ext, lat_ext)
            is_small_eddy = d.max() < d_thresh

    # Detected eddies:
            if eddy_area_within_limits * has_internal_ext * is_tall_eddy * is_small_eddy:
                # find centre of mass of eddy
                eddy_object_with_mass = field * region
                eddy_object_with_mass[np.isnan(eddy_object_with_mass)] = 0
                j_cen, i_cen = ndimage.center_of_mass(eddy_object_with_mass)
                lon_cen = np.interp(i_cen, range(0,len(lon)), lon)
                lat_cen = np.interp(j_cen, range(0,len(lat)), lat)
                lon_eddies = np.append(lon_eddies, lon_cen)
                lat_eddies = np.append(lat_eddies, lat_cen)
                # assign (and calculated) amplitude, area, and scale of eddies
                amp_eddies = np.append(amp_eddies, amp)
                area = region_Npix * res**2 * len_deg_lat * len_deg_lon(lat_cen) # [km**2]
                area_eddies = np.append(area_eddies, area)
                scale = np.sqrt(area / np.pi) # [km]
                scale_eddies = np.append(scale_eddies, scale)
                # remove its interior pixels from further eddy detection
                eddy_mask = np.ones(field.shape)
                eddy_mask[interior.astype(int)==1] = np.nan
                field = field * eddy_mask

    return lon_eddies, lat_eddies, amp_eddies, area_eddies, scale_eddies


def eddies_list(lon_eddies_a, lat_eddies_a, amp_eddies_a, area_eddies_a, scale_eddies_a, lon_eddies_c, lat_eddies_c, amp_eddies_c, area_eddies_c, scale_eddies_c):
    '''
    Creates list detected eddies
    '''

    eddies = []

    for ed in range(len(lon_eddies_c)):
        eddy_tmp = {}
        eddy_tmp['lon'] = np.append(lon_eddies_a[ed], lon_eddies_c[ed])
        eddy_tmp['lat'] = np.append(lat_eddies_a[ed], lat_eddies_c[ed])
        eddy_tmp['amp'] = np.append(amp_eddies_a[ed], amp_eddies_c[ed])
        eddy_tmp['area'] = np.append(area_eddies_a[ed], area_eddies_c[ed])
        eddy_tmp['scale'] = np.append(scale_eddies_a[ed], scale_eddies_c[ed])
        eddy_tmp['type'] = list(repeat('anticyclonic',len(lon_eddies_a[ed]))) + list(repeat('cyclonic',len(lon_eddies_c[ed])))
        eddy_tmp['N'] = len(eddy_tmp['lon'])
        eddies.append(eddy_tmp)

    return eddies


def eddies_init(det_eddies):
    '''
    Initializes list of eddies. The ith element of output is
    a dictionary of the ith eddy containing information about
    position and size as a function of time, as well as type.
    '''

    eddies = []

    for ed in range(det_eddies[0]['N']):
        eddy_tmp = {}
        eddy_tmp['lon'] = np.array([det_eddies[0]['lon'][ed]])
        eddy_tmp['lat'] = np.array([det_eddies[0]['lat'][ed]])
        eddy_tmp['amp'] = np.array([det_eddies[0]['amp'][ed]])
        eddy_tmp['area'] = np.array([det_eddies[0]['area'][ed]])
        eddy_tmp['scale'] = np.array([det_eddies[0]['scale'][ed]])
        eddy_tmp['type'] = det_eddies[0]['type'][ed]
        eddy_tmp['time'] = np.array([1])
        eddy_tmp['exist_at_start'] = True
        eddy_tmp['terminated'] = False
        eddies.append(eddy_tmp)

    return eddies


def load_rossrad():
    '''
    Load first baroclinic wave speed [m/s] and Rossby radius
    of deformation [km] data from rossrad.dat (Chelton et al., 1998)

    Also calculated is the first baroclinic Rossby wave speed [m/s]
    according to the formula:  cR = -beta rossby_rad**2
    '''

    data = np.loadtxt('data/rossrad.dat')

    rossrad = {}
    rossrad['lat'] = data[:,0]
    rossrad['lon'] = data[:,1]
    rossrad['c1'] = data[:,2] # m/s
    rossrad['rossby_rad'] = data[:,3] # km

    R = 6371.e3 # Radius of Earth [m]
    Sigma = 2 * np.pi / (24*60*60) # Rotation frequency of Earth [rad/s]
    beta = (2*Sigma/R) * np.cos(rossrad['lat']*np.pi/180) # 1 / m s
    rossrad['cR'] = -beta * (1e3*rossrad['rossby_rad'])**2

    return rossrad


def is_in_ellipse(x0, y0, dE, d, x, y):
    '''
    Check if point (x,y) is contained in ellipse given by the equation

      (x-x1)**2     (y-y1)**2
      ---------  +  ---------  =  1
         a**2          b**2

    where:

      a = 0.5 * (dE + dW)
      b = dE
      x1 = x0 + 0.5 * (dE - dW)
      y1 = y0
    '''

    dW = np.max([d, dE])

    b = dE
    a = 0.5 * (dE + dW)

    x1 = x0 + 0.5*(dE - dW)
    y1 = y0

    return (x-x1)**2 / a**2 + (y-y1)**2 / b**2 <= 1


def len_deg_lon(lat):
    '''
    Returns the length of one degree of longitude (at latitude
    specified) in km.
    '''

    R = 6371. # Radius of Earth [km]

    return (np.pi/180.) * R * np.cos( lat * np.pi/180. )


def calculate_d(dE, lon, lat, rossrad, dt):
    '''
    Calculates length of search area to the west of central point.
    This is equal to the length of the search area to the east of
    central point (dE) unless in the tropics ( abs(lat) < 18 deg )
    in which case the distance a Rossby wave travels in one time step
    (dt, days) is used instead.
    '''

    if np.abs(lat) < 18 :
        # Rossby wave speed [km/day]
        c = interpolate.griddata(np.array([rossrad['lon'], rossrad['lat']]).T, rossrad['cR'], (lon, lat), method='linear') * 86400. / 1000.
        d = np.abs(1.75 * c * dt)
    else:
        d = dE

    return d


def track_eddies(eddies, det_eddies, tt, dt, dt_aviso, dE_aviso, rossrad, eddy_scale_min, eddy_scale_max):
    '''
    Given a map of detected eddies as a function of time (det_eddies)
    this function will update tracks of individual eddies at time step
    tt in variable eddies
    '''

    # List of unassigned eddies at time tt

    unassigned = range(det_eddies[tt]['N'])

    # For each existing eddy (t<tt) loop through unassigned eddies and assign to existing eddy if appropriate

    for ed in range(len(eddies)):

        # Check if eddy has already been terminated

        if not eddies[ed]['terminated']:

            # Define search region around centroid of existing eddy ed at last known position
    
            x0 = eddies[ed]['lon'][-1] # [deg. lon]
            y0 = eddies[ed]['lat'][-1] # [deg. lat]
            dE = dE_aviso/(dt_aviso/dt) # [km]
            d = calculate_d(dE, x0, y0, rossrad, dt) # [km]
    
            # Find all eddy centroids in search region at time tt
    
            is_near = is_in_ellipse(x0, y0, dE/len_deg_lon(y0), d/len_deg_lon(y0), det_eddies[tt]['lon'][unassigned], det_eddies[tt]['lat'][unassigned])
    
            # Check if eddies' amp  and area are between 0.25 and 2.5 of original eddy
    
            amp = eddies[ed]['amp'][-1]
            area = eddies[ed]['area'][-1]
            is_similar_amp = (det_eddies[tt]['amp'][unassigned] < amp*eddy_scale_max) * (det_eddies[tt]['amp'][unassigned] > amp*eddy_scale_min)
            is_similar_area = (det_eddies[tt]['area'][unassigned] < area*eddy_scale_max) * (det_eddies[tt]['area'][unassigned] > area*eddy_scale_min)
    
            # Check if eddies' type is the same as original eddy
    
            is_same_type = np.array([det_eddies[tt]['type'][i] == eddies[ed]['type'] for i in unassigned])
    
            # Possible eddies are those which are near, of the right amplitude, and of the same type
    
            possibles = is_near * is_similar_amp * is_similar_area * is_same_type
            if possibles.sum() > 0:
    
                # Of all found eddies, accept only the nearest one
    
                dist = np.sqrt((x0-det_eddies[tt]['lon'][unassigned])**2 + (y0-det_eddies[tt]['lat'][unassigned])**2)
                nearest = dist == dist[possibles].min()
                next_eddy = unassigned[np.where(nearest * possibles)[0][0]]
    
                # Add coordinatse and properties of accepted eddy to trajectory of eddy ed
    
                eddies[ed]['lon'] = np.append(eddies[ed]['lon'], det_eddies[tt]['lon'][next_eddy])
                eddies[ed]['lat'] = np.append(eddies[ed]['lat'], det_eddies[tt]['lat'][next_eddy])
                eddies[ed]['amp'] = np.append(eddies[ed]['amp'], det_eddies[tt]['amp'][next_eddy])
                eddies[ed]['area'] = np.append(eddies[ed]['area'], det_eddies[tt]['area'][next_eddy])
                eddies[ed]['scale'] = np.append(eddies[ed]['scale'], det_eddies[tt]['scale'][next_eddy])
                eddies[ed]['time'] = np.append(eddies[ed]['time'], tt+1)
    
                # Remove detected eddy from list of eddies available for assigment to existing trajectories
    
                unassigned.remove(next_eddy)

            # Terminate eddy otherwise

            else:

                eddies[ed]['terminated'] = True

    # Create "new eddies" from list of eddies not assigned to existing trajectories

    if len(unassigned) > 0:

        for un in unassigned:

            eddy_tmp = {}
            eddy_tmp['lon'] = np.array([det_eddies[tt]['lon'][un]])
            eddy_tmp['lat'] = np.array([det_eddies[tt]['lat'][un]])
            eddy_tmp['amp'] = np.array([det_eddies[tt]['amp'][un]])
            eddy_tmp['area'] = np.array([det_eddies[tt]['area'][un]])
            eddy_tmp['scale'] = np.array([det_eddies[tt]['scale'][un]])
            eddy_tmp['type'] = det_eddies[tt]['type'][un]
            eddy_tmp['time'] = np.array([tt+1])
            eddy_tmp['exist_at_start'] = False
            eddy_tmp['terminated'] = False
            eddies.append(eddy_tmp)

    return eddies

