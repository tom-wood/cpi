# cpi/cpi11 (Contour Plotting for ISIS/I11)
CPI and CPI11 are programs defining a Dataset class (and some others) for
use in plotting and manipulating data collected at the diffraction beamlines
at ISIS (currently Gem and Polaris, although can be extended to others) and
I11 at Diamond. 

## Getting started
The ipython notebooks ```cpi_notebook.ipynb``` and 
```cpi11_notebook.ipynb``` give basic examples of how each flavour works.
You will need a functioning installation of python including numpy, scipy,
matplotlib, pandas (cpi and cpi11 have been tested with canopy and 
anaconda python distributions).

## Description of classes/functions (cpi)
Documentation on all classes/methods can be called by the command
```help(class.method)```.
### PlotDefaults class
This is a class for setting the matplotlib plotting defaults. Uset the 
```PlotDefaults.setp(param, value)``` method to set the parameters. 
Changeable parameters are found by ```help(PlotDefaults.setp)```.

### Dataset class
The Dataset class contains methods for getting, plotting and maniupulating
the data for ISIS beamlines. It is initialized by putting in 
```filepath```, ```first_file_number```, ```last_file_number``` and
```beamline``` arguments, as well as keyword arguments for ```beam_min```,
```bank_num``` and ```tof``` (see below for defaults).

#### Attributes
The Dataset class contains the following attributes:
1. ```filepath```: the filepath for the .dat and .log files.
2. ```expt_nums```: list of the experiment numbers.
3. ```beamline```: either ```'Polaris'``` or ```'Gem'```.
4. ```beam_min```: minimum average beam throughout run (default 100);
samples below this average are treated as beam offs.
5. ```bank_num```: the bank number (default 4).
6. ```tof```: boolean, ```False``` gives d-spacing, ```True``` gives
time of flight data.
7. ```scan_times```: list loaded when ```get_scan_times``` method is
invoked.
8. ```lstarts```: list of log file start times (i.e. first time recorded
in each log file; from ```get_scan_times``` method).
9. ```lends```: list of log file end times (i.e. last time recorded
in each log file; from ```get_scan_times``` method).
10. ```av_bcs```: list of average beam currents (in order to determine
whether data is treated as beam off or not (from ```get_scan_times``` 
method).
11. ```beam_offs```: list of beam offs (when average beam current is less
than ```beam_min``` (from ```get_scan_times``` method).
12. ```beam_offs2```: list of beam offs (when there has been a break in the
log files; from ```get_scan_times``` method).
13. ```Tvals```: list of temperature values for each run number (from 
```get_scan_times``` method).
14. ```data```: list of pandas Dataframes for each run with x, y and e
values (from ```get_data``` method).
15. ```igan_data```: array of data from the IGAn (from 
```get_igan_data``` method).

#### Methods
And the following methods:
1. ```get_scan_times()```: this method loads scan times from .log files
within ```Dataset.filepath```. If ```Tstring``` is given a value then 
temperature values will be read from the log files as well. Beam offs should
be easily dealt with using this method.
2. ```get_run_numbers()```: returns an array of run numbers (accounting for
beam off periods).
3. ```get_expt_fnames_all()```: returns a list of all experiment file names 
(including missing ones).
4. ```get_expt_fnames()```: returns a list of experiment file names 
excluding missing ones).
5. ```get_data()```: fetches diffraction data from .dat files.
6. ```get_igan_data()```: if the IGAn equipment has been used, then this
method can fetch the relevant data.
7. ```data_xy()```: returns two arrays of x data and y data.
8. ```get_indices()```: returns list of indices for the diffraction data
given run number range and y range.
9. ```get_max_intensity()```: returns the maximum intensity recorded within
a certain range.
10. ```get_min_intensity()```: returns the minimum intensity recorded within
a certain range.
11. ```get_max_intensities()```: returns array of the maximum intensity for
each run within a certain range (will include y values and run numbers by
default).
12. ```get_min_intensities()```: returns array of the minimum intensity for
each run within a certain range (will include y values and run numbers by
default).
13. ```to_xye()```: writes xye files for all diffraction runs within a
Dataset (particularly useful combined with ```sum_dsets()``` method).
14. ```sum_dsets()```: returns summed (mean) of diffraction runs within 
specifiied range.
15. ```plot()```: Returns a 2D plot of a diffraction run (specified by 
```tval``` argument).
16. ```get_deltad_over_d()```: Returns a list of delta d over d arrays given
a run and a range of d values (useful for comparing strain effects).
17. ```plot_deltad_over_d()```: Plots deltad over d for strain comparison.
18. ```plotQ()```: plots using Q rather than d.
19. ```contour_plot()```: will give a contour plot of the diffraction data.
20. ```contour_temp()```: requires ```T``` argument (should use 
```Dataset.T_vals``` if possible) and gives contour plot of the diffraction
data with a plot of temperature above it.
21. ```contour_igan()```: will give a contour plot along with the IGAn 
pressure, mass and temperature readings.
22. ```contour_mult()```: will plot multiple contour plots (with shared x
 and y axes if wanted), enabling zooming in on certain regions (for 
example).
23. ```plot_animate()```: will produce an animation cycling through the 2D
diffraction plots (requires a time range argument to be set).
24. ```contour_animate()```: will produce an animation cycling through the
2D diffraction plots along with the contour plot (and the temperature).

### Other functions
These are mostly (except for ```plotQ```) used for debugging purposes.
1. ```get_expt_numbers```: returns list of experiment numbers.
2. ```get_file_list```: returns list of files at filepath.
3. ```get_expt_fnames_all```: returns list of all dataset filenames.
4. ```get_expt_fnames```: returns list of all dataset filenames (excluding
beam offs).
5. ```plotQ```: can plot multiple datasets in Q (for example when comparing
different banks).

## Differences with cpi11 
The PlotDefaults class remains unchanged. The Dataset class is now 
initialized without ```beamline```, ```beam_min```, ```bank_num``` and
```tof``` arguments and instead with keyword arguments of ```detector```, 
```mac_suffix```, ```psd_suffix```, ```log_fname```, ```wavelength``` and
```zpe``` (see below for more details

### Dataset differences
#### Attributes missing
These are: ```beamline```, ```beam_min```, ```bank_num```, ```tof```,
```scan_times```, ```lstarts```, ```lends```, ```av_bcs```, 
```beam_offs2``` and ```igan_data```.

#### New attributes
1. ```detector```: either ```'mac'``` or ```'psd'```.
2. ```mac_suffix```: string for mac filenames.
3. ```psd_suffix```: string for psd filenames.
4. ```wavelength```: wavelength in Angstroms.
5. ```zpe```: zero error for 2theta.

#### Methods missing
These are: ```get_scan_times()``` and ```get_igan_data()```.

#### New methods
1. ```get_temps()```: Fetches temperatures for each run by reading all log
files (by default).
2. ```twotheta_to_d()```: returns a new dataset but with two theta x values
as d values calculated according to wavelength and zero-error (note that
d values will be in descending order).
3. ```plot_reflected_peak()```: plot method for observing asymmetric peaks,
by plotting the reflection of a peak against itself.

### Other differences
There are no extra functions in cpi11.
