'''

  Plot eddy tracks

'''

# Load required modules

import numpy as np
from matplotlib import pyplot as plt
import mpl_toolkits.basemap as bm

import params

# Load eddy data

data_dir = './'
data_dir = params.data_dir

#run = 'cb_AVISO'
#data = np.load(data_dir+'eddy_track_'+run+'.npz')
#eddies_AVISO = data['eddies']

run = 'cb_NEMO'
data = np.load(data_dir+'eddy_track_'+run+'.npz')
eddies_CTRL = data['eddies']

#run = 'CTRL'
#data = np.load(data_dir+'eddy_track_'+run+'.npz')
#eddies_CTRL = data['eddies']

#run = 'A1B'
#data = np.load(data_dir+'eddy_track_'+run+'.npz')
#eddies_A1B = data['eddies']

# Plot eddy tracks

age_min_weeks = [4, 16, 32]
#age_min_weeks = [8, 32]

domain = [134, 180, -55, 0]
#domain = [90, 180, -55, 0]
plt.figure()

cnt = 1
for age in age_min_weeks:
#
    age_min = age*7 # [days]
#
    plt.subplot(3, len(age_min_weeks), cnt)
    proj = bm.Basemap(projection='merc', llcrnrlat=domain[2], llcrnrlon=domain[0], urcrnrlat=domain[3], urcrnrlon=domain[1], resolution='i')
    proj.drawcoastlines()
    proj.drawparallels(range(-50,0+1,10), labels=[True,False,False,False])
    proj.drawmeridians(range(140,180+1,10), labels=[False,False,False,True])
    plt.title('Eddy age > ' + str(age) + ' weeks')

    for ed in range(len(eddies_CTRL)):
        #print ed
        lon, lat = proj(eddies_CTRL[ed]['lon'], eddies_CTRL[ed]['lat'])
        if (eddies_CTRL[ed]['age'] > age_min) & (eddies_CTRL[ed]['type'] == 'anticyclonic'):
            plt.plot(lon, lat, 'r-')
            plt.plot(lon[-1], lat[-1], 'ro')
        if (eddies_CTRL[ed]['age'] > age_min) & (eddies_CTRL[ed]['type'] == 'cyclonic'):
            plt.plot(lon, lat, 'b-')
            plt.plot(lon[-1], lat[-1], 'bo')

    plt.subplot(3, len(age_min_weeks), cnt+len(age_min_weeks))
    proj = bm.Basemap(projection='merc', llcrnrlat=domain[2], llcrnrlon=domain[0], urcrnrlat=domain[3], urcrnrlon=domain[1], resolution='i')
    proj.drawcoastlines()
    proj.drawparallels(range(-50,0+1,10), labels=[True,False,False,False])
    proj.drawmeridians(range(140,180+1,10), labels=[False,False,False,True])
    plt.title('Eddy age > ' + str(age) + ' weeks')
#
    #for ed in range(len(eddies_A1B)):
        #lon, lat = proj(eddies_A1B[ed]['lon'], eddies_A1B[ed]['lat'])
        #if (eddies_A1B[ed]['age'] > age_min) & (eddies_A1B[ed]['type'] == 'anticyclonic'):
            #plt.plot(lon, lat, 'r-')
            #plt.plot(lon[-1], lat[-1], 'ro')
        #if (eddies_A1B[ed]['age'] > age_min) & (eddies_A1B[ed]['type'] == 'cyclonic'):
            #plt.plot(lon, lat, 'b-')
            #plt.plot(lon[-1], lat[-1], 'bo')
#
    plt.subplot(3, len(age_min_weeks), cnt+2*len(age_min_weeks))
    proj = bm.Basemap(projection='merc', llcrnrlat=domain[2], llcrnrlon=domain[0], urcrnrlat=domain[3], urcrnrlon=domain[1], resolution='i')
    proj.drawcoastlines()
    proj.drawparallels(range(-50,0+1,10), labels=[True,False,False,False])
    proj.drawmeridians(range(140,180+1,10), labels=[False,False,False,True])
    plt.title('Eddy age > ' + str(age) + ' weeks')
#
    #for ed in range(len(eddies_AVISO)):
        #lon, lat = proj(eddies_AVISO[ed]['lon'], eddies_AVISO[ed]['lat'])
        #if (eddies_AVISO[ed]['age']*7 > age_min) & (eddies_AVISO[ed]['type'] == 'anticyclonic'):
            #plt.plot(lon, lat, 'r-')
            #plt.plot(lon[-1], lat[-1], 'ro')
        #if (eddies_AVISO[ed]['age']*7 > age_min) & (eddies_AVISO[ed]['type'] == 'cyclonic'):
            #plt.plot(lon, lat, 'b-')
            #plt.plot(lon[-1], lat[-1], 'bo')
#
    cnt += 1


#plt.show()
#plt.savefig('../../../documents/10_Tasman_Sea_Eddies/figures/eddies_OFAM_orig.pdf', bbox_inches='tight', pad_inches=0.5)
#plt.savefig('./eddies_OFAM_orig1.png', bbox_inches='tight', pad_inches=0.5)
plt.savefig(params.plot_dir+'eddies_OFAM_orig1.png', bbox_inches='tight', pad_inches=0.5)

domain = [134, 180, -50, -25]
plt.figure()
cnt = 1
for age in age_min_weeks:
#
    age_min = age*7 # [days]
#
    plt.subplot(2, len(age_min_weeks), cnt)
    proj = bm.Basemap(projection='merc', llcrnrlat=domain[2], llcrnrlon=domain[0], urcrnrlat=domain[3], urcrnrlon=domain[1], resolution='i')
    proj.drawcoastlines()
    proj.drawparallels(range(-50,0+1,10), labels=[True,False,False,False])
    proj.drawmeridians(range(140,180+1,10), labels=[False,False,False,True])
    plt.title('Eddy age > ' + str(age) + ' weeks')
#
    for ed in range(len(eddies_CTRL)):
        lon, lat = proj(eddies_CTRL[ed]['lon'], eddies_CTRL[ed]['lat'])
        if (eddies_CTRL[ed]['age'] > age_min) & (eddies_CTRL[ed]['type'] == 'anticyclonic'):
            plt.plot(lon, lat, 'r-')
            plt.plot(lon[-1], lat[-1], 'ro')
        if (eddies_CTRL[ed]['age'] > age_min) & (eddies_CTRL[ed]['type'] == 'cyclonic'):
            plt.plot(lon, lat, 'b-')
            plt.plot(lon[-1], lat[-1], 'bo')
#
    cnt += 1
#
    plt.subplot(2, len(age_min_weeks), cnt)
    proj = bm.Basemap(projection='merc', llcrnrlat=domain[2], llcrnrlon=domain[0], urcrnrlat=domain[3], urcrnrlon=domain[1], resolution='i')
    proj.drawcoastlines()
    proj.drawparallels(range(-50,0+1,10), labels=[True,False,False,False])
    proj.drawmeridians(range(140,180+1,10), labels=[False,False,False,True])
    plt.title('Eddy age > ' + str(age) + ' weeks')
#
    #for ed in range(len(eddies_A1B)):
        #lon, lat = proj(eddies_A1B[ed]['lon'], eddies_A1B[ed]['lat'])
        #if (eddies_A1B[ed]['age'] > age_min) & (eddies_A1B[ed]['type'] == 'anticyclonic'):
            #plt.plot(lon, lat, 'r-')
            #plt.plot(lon[-1], lat[-1], 'ro')
        #if (eddies_A1B[ed]['age'] > age_min) & (eddies_A1B[ed]['type'] == 'cyclonic'):
            #plt.plot(lon, lat, 'b-')
            #plt.plot(lon[-1], lat[-1], 'bo')
#
    cnt += 1

# plt.savefig('../../../documents/10_Tasman_Sea_Eddies/figures/eddies_OFAM_orig.pdf', bbox_inches='tight', pad_inches=0.5)
# plt.savefig('../../../documents/10_Tasman_Sea_Eddies/figures/eddies_OFAM_orig.png', bbox_inches='tight', pad_inches=0.5)
#plt.savefig('./eddies_OFAM_orig2.png', bbox_inches='tight', pad_inches=0.5)
plt.savefig(params.plot_dir+'eddies_OFAM_orig2.png', bbox_inches='tight', pad_inches=0.5)


# Plot eddy generation locations

age_min = 32*7 # days
age_min = 16*7 # days

plt.figure()

plt.subplot(1, 2, 1)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[2], llcrnrlon=domain[0], urcrnrlat=domain[3], urcrnrlon=domain[1], resolution='i')
proj.drawcoastlines()
proj.drawparallels(range(-50,0+1,10), labels=[True,False,False,False])
proj.drawmeridians(range(140,180+1,10), labels=[False,False,False,True])
plt.title('Eddy generation locations (CTRL)')

for ed in range(len(eddies_CTRL)):
    lon, lat = proj(eddies_CTRL[ed]['lon'], eddies_CTRL[ed]['lat'])
    if (eddies_CTRL[ed]['age'] > age_min) & (eddies_CTRL[ed]['type'] == 'anticyclonic'):
            plt.plot(lon[0], lat[0], 'ro')
            plt.plot(lon[-1], lat[-1], 'rx')
    if (eddies_CTRL[ed]['age'] > age_min) & (eddies_CTRL[ed]['type'] == 'cyclonic'):
            plt.plot(lon[0], lat[0], 'bo')
            plt.plot(lon[-1], lat[-1], 'bx')

plt.subplot(1, 2, 2)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[2], llcrnrlon=domain[0], urcrnrlat=domain[3], urcrnrlon=domain[1], resolution='i')
proj.drawcoastlines()
proj.drawparallels(range(-50,0+1,10), labels=[True,False,False,False])
proj.drawmeridians(range(140,180+1,10), labels=[False,False,False,True])
plt.title('Eddy generation locations (A1B)')

#for ed in range(len(eddies_A1B)):
    #lon, lat = proj(eddies_A1B[ed]['lon'], eddies_A1B[ed]['lat'])
    #if (eddies_A1B[ed]['age'] > age_min) & (eddies_A1B[ed]['type'] == 'anticyclonic'):
            #plt.plot(lon[0], lat[0], 'ro')
            #plt.plot(lon[-1], lat[-1], 'rx')
    #if (eddies_A1B[ed]['age'] > age_min) & (eddies_A1B[ed]['type'] == 'cyclonic'):
            #plt.plot(lon[0], lat[0], 'bo')
            #plt.plot(lon[-1], lat[-1], 'bx')
plt.savefig(params.plot_dir+'last.png')
