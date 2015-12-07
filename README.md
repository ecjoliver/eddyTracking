
# Automated eddy detection
 
Eric Oliver

Code Description
============                     

eddy_detection.py    - Code for the detection of eddies given a series of sea level maps

eddy_tracking.py     - Code for the tracking of eddies after detection has been performed

params.py            - Parameter file used by eddy detection and eddy tracking programs

eddy_census.py       - Code for calculating census statistics of tracked eddies

eddy_plot.py         - Code for plotting eddy tracks

eddy_functions.py    - Module of supporting functions

Notes
============                     

1. This code as been applied to model output from OFAM  and from weekly and daily sea level maps from Aviso. To apply it to another dataset it is necessary to make the following adjustments:

 a. Add a new dataset id 'NAME' in the conditionals in params.py, including relevant parameters (number of time steps, resolution, time step).

 b. Add appropriate code in 'load_lonlat' and 'load_eta' functions (both in eddy_functions.py) to properly handle the loading of your data. Code assumes one file (spatial map) per time step.


Changes added by Christopher Bull (Dec 2015)
============                     

Modified to work with NEMO. Made some general changes including:

 1. Works with python netCDF4 library.
 1. Refactored so global data and plot dirs are defined in params.py (and created if they don't exist)
 1. Refactored so that pathroot for input data can be defined in params rather than eddy_functions.
 1. Added quick_plot and detection_plot functions to eddy_function.py
 1. Added functions from ecoliver (that were not previously included) into eddy_functions.py
 1. Added rossrad.dat
