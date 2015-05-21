'''

  Calculate eddy census statistics
  for tracked eddies

'''

# Load required modules

import numpy as np
from matplotlib import pyplot as plt
import mpl_toolkits.basemap as bm


# Load eddy data

data_dir = '/home/ecoliver/Desktop/data/eddy/'
run = 'AVISO'
data = np.load(data_dir+'eddy_track_'+run+'.npz')
eddies_AVISO = data['eddies']
run = 'CTRL'
data = np.load(data_dir+'eddy_track_'+run+'.npz')
eddies_CTRL = data['eddies']
run = 'A1B'
data = np.load(data_dir+'eddy_track_'+run+'.npz')
eddies_A1B = data['eddies']

# Some variables

age_min_weeks = 16 # [weeks]
age_min = 7*age_min_weeks # [days]

lon = np.arange(90, 170+1, 1)
lat = np.arange(-50, -25+1, 1)
llon, llat = np.meshgrid(lon, lat)
X = len(lon)
Y = len(lat)
DIM = (Y,X)

# Calculate statistics (AVISO)

N_AVISO = np.zeros(DIM)
Nc_unique_AVISO = np.zeros(DIM)
Na_unique_AVISO = np.zeros(DIM)
counted_AVISO = -1*np.ones(DIM, dtype=np.object)
amp_AVISO = np.zeros(DIM)
scale_AVISO = np.zeros(DIM)

for ed in range(len(eddies_AVISO)):
    print ed, len(eddies_AVISO)
#
    if (eddies_AVISO[ed]['age'] >= age_min_weeks ):
#
        for t in range(eddies_AVISO[ed]['age']):
#
            i = np.where((lon > eddies_AVISO[ed]['lon'][t]-1) * (lon < eddies_AVISO[ed]['lon'][t]))[0]
            j = np.where((lat > eddies_AVISO[ed]['lat'][t]-1) * (lat < eddies_AVISO[ed]['lat'][t]))[0]
            if len(j)>0 and len(i)>0:
                amp_AVISO[j,i] += eddies_AVISO[ed]['amp'][t]
                scale_AVISO[j,i] += eddies_AVISO[ed]['scale'][t]
                N_AVISO[j,i] += 1
#
                firstcount = type(counted_AVISO[j[0],i[0]])==type(-1)
                notcountedyet = False
                if not firstcount:
                    notcountedyet = not (ed in counted_AVISO[j[0],i[0]])
                if firstcount or notcountedyet:
                    if eddies_AVISO[ed]['type'] == 'anticyclonic':
                        Na_unique_AVISO[j,i] += 1
                    else:
                        Nc_unique_AVISO[j,i] += 1
                    counted_AVISO[j[0],i[0]] = np.append(counted_AVISO[j[0],i[0]], ed)

N_unique_AVISO = Nc_unique_AVISO + Na_unique_AVISO
cyc_AVISO = Nc_unique_AVISO / Na_unique_AVISO
pcyc_AVISO = Nc_unique_AVISO / N_unique_AVISO
amp_AVISO /= N_AVISO
scale_AVISO /= N_AVISO

# Save

np.savez(data_dir+'eddy_census_AVISO', age_min=age_min, age_min_weeks=age_min_weeks, llon=llon, llat=llat, N_unique_AVISO=N_unique_AVISO, Nc_unique_AVISO=Nc_unique_AVISO, Na_unique_AVISO=Na_unique_AVISO, cyc_AVISO=cyc_AVISO, pcyc_AVISO=pcyc_AVISO, amp_AVISO=amp_AVISO, scale_AVISO=scale_AVISO)

# Calculate statistics (CTRL)

N_CTRL = np.zeros(DIM)
Nc_unique_CTRL = np.zeros(DIM)
Na_unique_CTRL = np.zeros(DIM)
counted_CTRL = -1*np.ones(DIM, dtype=np.object)
amp_CTRL = np.zeros(DIM)
scale_CTRL = np.zeros(DIM)

for ed in range(len(eddies_CTRL)):
    print ed, len(eddies_CTRL)
#
    if (eddies_CTRL[ed]['age'] >= age_min ):
#
        for t in range(eddies_CTRL[ed]['age']):
#
            i = np.where((lon > eddies_CTRL[ed]['lon'][t]-1) * (lon < eddies_CTRL[ed]['lon'][t]))[0]
            j = np.where((lat > eddies_CTRL[ed]['lat'][t]-1) * (lat < eddies_CTRL[ed]['lat'][t]))[0]
            if len(j)>0 and len(i)>0:
                amp_CTRL[j,i] += eddies_CTRL[ed]['amp'][t]
                scale_CTRL[j,i] += eddies_CTRL[ed]['scale'][t]
                N_CTRL[j,i] += 1
#
                firstcount = type(counted_CTRL[j[0],i[0]])==type(-1)
                notcountedyet = False
                if not firstcount:
                    notcountedyet = not (ed in counted_CTRL[j[0],i[0]])
                if firstcount or notcountedyet:
                    if eddies_CTRL[ed]['type'] == 'anticyclonic':
                        Na_unique_CTRL[j,i] += 1
                    else:
                        Nc_unique_CTRL[j,i] += 1
                    counted_CTRL[j[0],i[0]] = np.append(counted_CTRL[j[0],i[0]], ed)

N_unique_CTRL = Nc_unique_CTRL + Na_unique_CTRL
cyc_CTRL = Nc_unique_CTRL / Na_unique_CTRL
pcyc_CTRL = Nc_unique_CTRL / N_unique_CTRL
amp_CTRL /= N_CTRL
scale_CTRL /= N_CTRL

# Calculate statistics (A1B)

N_A1B = np.zeros(DIM)
Nc_unique_A1B = np.zeros(DIM)
Na_unique_A1B = np.zeros(DIM)
counted_A1B = -1*np.ones(DIM, dtype=np.object)
amp_A1B = np.zeros(DIM)
scale_A1B = np.zeros(DIM)

for ed in range(len(eddies_A1B)):
    print ed, len(eddies_A1B)
#
    if (eddies_A1B[ed]['age'] >= age_min ):
#
        for t in range(eddies_A1B[ed]['age']):
#
            i = np.where((lon > eddies_A1B[ed]['lon'][t]-1) * (lon < eddies_A1B[ed]['lon'][t]))[0]
            j = np.where((lat > eddies_A1B[ed]['lat'][t]-1) * (lat < eddies_A1B[ed]['lat'][t]))[0]
            if len(j)>0 and len(i)>0:
                amp_A1B[j,i] += eddies_A1B[ed]['amp'][t]
                scale_A1B[j,i] += eddies_A1B[ed]['scale'][t]
                N_A1B[j,i] += 1
#
                firstcount = type(counted_A1B[j[0],i[0]])==type(-1)
                notcountedyet = False
                if not firstcount:
                    notcountedyet = not (ed in counted_A1B[j[0],i[0]])
                if firstcount or notcountedyet:
                    if eddies_A1B[ed]['type'] == 'anticyclonic':
                        Na_unique_A1B[j,i] += 1
                    else:
                        Nc_unique_A1B[j,i] += 1
                    counted_A1B[j[0],i[0]] = np.append(counted_A1B[j[0],i[0]], ed)

N_unique_A1B = Nc_unique_A1B + Na_unique_A1B
cyc_A1B = Nc_unique_A1B / Na_unique_A1B
pcyc_A1B = Nc_unique_A1B / N_unique_A1B
amp_A1B /= N_A1B
scale_A1B /= N_A1B

# Save

np.savez(data_dir+'eddy_census_OFAM', age_min=age_min, age_min_weeks=age_min_weeks, llon=llon, llat=llat, N_unique_CTRL=N_unique_CTRL, Nc_unique_CTRL=Nc_unique_CTRL, Na_unique_CTRL=Na_unique_CTRL, cyc_CTRL=cyc_CTRL, pcyc_CTRL=pcyc_CTRL, amp_CTRL=amp_CTRL, scale_CTRL=scale_CTRL, N_unique_A1B=N_unique_A1B, Nc_unique_A1B=Nc_unique_A1B, Na_unique_A1B=Na_unique_A1B, cyc_A1B=cyc_A1B, pcyc_A1B=pcyc_A1B, amp_A1B=amp_A1B, scale_A1B=scale_A1B)

# Plot

domain = [134, 180, -55, 0]
domain = [90, 180, -55, 0]

plt.figure()
plt.subplot(1,3,1)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[2], llcrnrlon=domain[0], urcrnrlat=domain[3], urcrnrlon=domain[1], resolution='i')
proj.drawcoastlines()
proj.drawparallels(range(-50,0+1,10), labels=[True,False,False,False])
proj.drawmeridians(range(140,180+1,10), labels=[False,False,False,True])
plt.title('Eddy count - age > ' + str(age_min_weeks) + ' weeks - max = ' + str(np.nanmax(N_unique_CTRL)))
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, 1.*N_unique_CTRL / np.nanmax(N_unique_CTRL), levels=np.arange(0, 1+0.1, 0.1))
plt.clim(0, 1)
plt.colorbar()
plt.subplot(1,3,2)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[2], llcrnrlon=domain[0], urcrnrlat=domain[3], urcrnrlon=domain[1], resolution='i')
proj.drawcoastlines()
proj.drawparallels(range(-50,0+1,10), labels=[True,False,False,False])
proj.drawmeridians(range(140,180+1,10), labels=[False,False,False,True])
plt.title('Eddy count - age > ' + str(age_min_weeks) + ' weeks - max = ' + str(np.nanmax(N_unique_A1B)))
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, 1.*N_unique_A1B / np.nanmax(N_unique_A1B), levels=np.arange(0, 1+0.1, 0.1))
plt.clim(0, 1)
plt.colorbar()
plt.subplot(1,3,3)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[2], llcrnrlon=domain[0], urcrnrlat=domain[3], urcrnrlon=domain[1], resolution='i')
proj.drawcoastlines()
proj.drawparallels(range(-50,0+1,10), labels=[True,False,False,False])
proj.drawmeridians(range(140,180+1,10), labels=[False,False,False,True])
plt.title('Difference')
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, N_unique_A1B -N_unique_CTRL)
plt.contour(lonproj, latproj, N_unique_A1B -N_unique_CTRL, levels=[0], colors='k')
plt.colorbar()

plt.figure()
plt.subplot(1,2,1)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[2], llcrnrlon=domain[0], urcrnrlat=domain[3], urcrnrlon=domain[1], resolution='i')
proj.drawcoastlines()
proj.drawparallels(range(-50,0+1,10), labels=[True,False,False,False])
proj.drawmeridians(range(140,180+1,10), labels=[False,False,False,True])
plt.title('Ratio of cyc. to anticyc. eddies (log10 scale)')
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, np.log10(cyc_CTRL), levels=np.arange(-2, 2+0.1, 0.1), cmap=plt.cm.RdBu_r)
plt.clim(-2, 2)
plt.colorbar()
plt.subplot(1,2,2)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[2], llcrnrlon=domain[0], urcrnrlat=domain[3], urcrnrlon=domain[1], resolution='i')
proj.drawcoastlines()
proj.drawparallels(range(-50,0+1,10), labels=[True,False,False,False])
proj.drawmeridians(range(140,180+1,10), labels=[False,False,False,True])
plt.title('Ratio of cyc. to anticyc. eddies (log10 scale)')
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, np.log10(cyc_A1B), levels=np.arange(-2, 2+0.1, 0.1), cmap=plt.cm.RdBu_r)
plt.clim(-2, 2)
plt.colorbar()

plt.figure()
plt.subplot(1,2,1)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[2], llcrnrlon=domain[0], urcrnrlat=domain[3], urcrnrlon=domain[1], resolution='i')
proj.drawcoastlines()
proj.drawparallels(range(-50,0+1,10), labels=[True,False,False,False])
proj.drawmeridians(range(140,180+1,10), labels=[False,False,False,True])
plt.title('Proportion of cyc. to anticyc. eddies (log10 scale)')
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, pcyc_CTRL, levels=np.arange(0, 1+0.1, 0.1), cmap=plt.cm.RdBu)
plt.clim(0, 1)
plt.colorbar()
plt.subplot(1,2,2)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[2], llcrnrlon=domain[0], urcrnrlat=domain[3], urcrnrlon=domain[1], resolution='i')
proj.drawcoastlines()
proj.drawparallels(range(-50,0+1,10), labels=[True,False,False,False])
proj.drawmeridians(range(140,180+1,10), labels=[False,False,False,True])
plt.title('Proportion of cyc. to anticyc. eddies (log10 scale)')
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, pcyc_A1B, levels=np.arange(0, 1+0.1, 0.1), cmap=plt.cm.RdBu)
plt.clim(0, 1)
plt.colorbar()
# plt.savefig('../../../documents/06_EAC_sep_and_EKE/figures/pcyc.pdf', bbox_inches='tight', pad_inches=0.5)

plt.figure()
plt.subplot(1,3,1)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[2], llcrnrlon=domain[0], urcrnrlat=domain[3], urcrnrlon=domain[1], resolution='i')
proj.drawcoastlines()
proj.drawparallels(range(-50,0+1,10), labels=[True,False,False,False])
proj.drawmeridians(range(140,180+1,10), labels=[False,False,False,True])
plt.title('Eddy amplitude [cm]')
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, 100*amp_CTRL) #, levels=np.arange(0, 1+0.1, 0.1))
plt.colorbar()
plt.subplot(1,3,2)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[2], llcrnrlon=domain[0], urcrnrlat=domain[3], urcrnrlon=domain[1], resolution='i')
proj.drawcoastlines()
proj.drawparallels(range(-50,0+1,10), labels=[True,False,False,False])
proj.drawmeridians(range(140,180+1,10), labels=[False,False,False,True])
plt.title('Eddy amplitude [cm]')
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, 100*amp_A1B)
plt.colorbar()
plt.subplot(1,3,3)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[2], llcrnrlon=domain[0], urcrnrlat=domain[3], urcrnrlon=domain[1], resolution='i')
proj.drawcoastlines()
proj.drawparallels(range(-50,0+1,10), labels=[True,False,False,False])
proj.drawmeridians(range(140,180+1,10), labels=[False,False,False,True])
plt.title('Difference')
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, 100*(amp_A1B-amp_CTRL))
plt.contour(lonproj, latproj, 100*(amp_A1B-amp_CTRL), levels=[0], colors='k')
plt.colorbar()

plt.figure()
plt.subplot(1,3,1)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[2], llcrnrlon=domain[0], urcrnrlat=domain[3], urcrnrlon=domain[1], resolution='i')
proj.drawcoastlines()
proj.drawparallels(range(-50,0+1,10), labels=[True,False,False,False])
proj.drawmeridians(range(140,180+1,10), labels=[False,False,False,True])
plt.title('Eddy scale [km]')
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, scale_CTRL) #, levels=np.arange(0, 1+0.1, 0.1))
plt.colorbar()
plt.subplot(1,3,2)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[2], llcrnrlon=domain[0], urcrnrlat=domain[3], urcrnrlon=domain[1], resolution='i')
proj.drawcoastlines()
proj.drawparallels(range(-50,0+1,10), labels=[True,False,False,False])
proj.drawmeridians(range(140,180+1,10), labels=[False,False,False,True])
plt.title('Eddy scale [km]')
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, scale_A1B) #, levels=np.arange(0, 1+0.1, 0.1))
plt.colorbar()
plt.subplot(1,3,3)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[2], llcrnrlon=domain[0], urcrnrlat=domain[3], urcrnrlon=domain[1], resolution='i')
proj.drawcoastlines()
proj.drawparallels(range(-50,0+1,10), labels=[True,False,False,False])
proj.drawmeridians(range(140,180+1,10), labels=[False,False,False,True])
plt.title('Difference')
lonproj, latproj = proj(llon, llat)
plt.contourf(lonproj, latproj, scale_A1B-scale_CTRL) #, levels=np.arange(0, 1+0.1, 0.1))
plt.contour(lonproj, latproj, scale_A1B-scale_CTRL, levels=[0], colors='k')
plt.colorbar()

plt.figure()
plt.subplot(1,2,1)
plt.plot( scale_CTRL.flatten(), 100*amp_CTRL.flatten(), 'ko')
plt.xlabel('scale [km]')
plt.ylabel('amplitude [cm]')
plt.xlim(35, 75)
plt.ylim(0, 14)
plt.subplot(1,2,2)
plt.plot( scale_A1B.flatten(), 100*amp_A1B.flatten(), 'ko')
plt.xlabel('scale [km]')
plt.xlim(35, 75)
plt.ylim(0, 14)

