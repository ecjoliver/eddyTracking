# stormTracking
 
Automated storm tracking software. Both cyclonic storms and anticyclonic high-pressure are detected and tracked.

Code Description
============                     

File                 |Description
---------------------|----------
|storm_detection.py    | Code for the detection of storms given a series of sea level maps|
|storm_tracking.py     | Code for the tracking of storms after detection has been performed|
|storm_census.py       | Code for calculating census statistics of tracked storms|
|storm_plot.py         | Code for plotting storm tracks|
|storm_functions.py    | Module of supporting functions|

## Notes

This code as been applied to 6-hourly mean sea level pressure maps from NCEP Twentieth Century Reanalysis. To apply it to another dataset it is necessary to edit the data-loading code near the top of eddy_detection.py as well as set function options as necessary (e.g. grid resolution, time step, etc).

## Contact                                                                                                          
Eric C. J. Oliver
Department of Oceanography
Dalhousie University
Halifax, Nova Scotia, Canada
e: eric.oliver@dal.ca
w: http://ecjoliver.weebly.com
https://github.com/ecjoliver
