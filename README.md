# CPI/CPI11 (Contour Plotting for ISIS/I11)
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

## Description of classes/functions (CPI)
### PlotDefaults class
This is a class for setting the matplotlib plotting defaults. Uset the 
```PlotDefaults.setp(param, value)``` method to set the parameters. 
Changeable parameters are found by ```help(PlotDefaults.setp)```.

### Dataset class
The Dataset class contains methods for getting, plotting and maniupulating
the data for ISIS beamlines.

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

And the following methods:

