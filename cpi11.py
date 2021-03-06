#Version 0.5.1-beta
#05/02/2019: added auto_offset_y kwarg to plot method
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd #used mainly for the pd.read_csv function (very fast)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes #for colour bars
from matplotlib import animation #for animation
import os #for finding out which files are in a directory
import fnmatch

mpl.rcParams['mathtext.default'] = 'regular'
mpl.rcParams['font.size'] = 16
mpl.rc('axes', labelsize=16)#, linewidth=2.0)
mpl.rc('ytick', labelsize=16)#, direction='out')
mpl.rc('xtick', labelsize=16)#, direction='out')
mpl.rc('xtick.major', size=6)#, width=2)
mpl.rc('ytick.major', size=6)#, width=2)
plt.rcParams['image.cmap'] = 'afmhot'

class PlotDefaults():
    def __init__(self):
        self.reset()
    def reset(self):
        mpl.rcParams['mathtext.default'] = 'regular'
        mpl.rcParams['font.size'] = 16
        mpl.rc('axes', labelsize=16)
        mpl.rc('ytick', labelsize=16)
        mpl.rc('xtick', labelsize=16)
        mpl.rc('xtick.major', size=6)
        mpl.rc('ytick.major', size=6)
        plt.rcParams['image.cmap'] = 'afmhot'
    def _setp(self, param, value):
        if param == 'mathtext.default':
            mpl.rcParams['mathtext.default'] = value
        elif param == 'font.size':
            mpl.rcParams['font.size'] = float(value)
        elif param == 'axes.labelsize':
            mpl.rc('axes', labelsize=float(value))
        elif param == 'axes.linewidth':
            mpl.rc('axes', linewidth=float(value))
        elif param == 'xtick.labelsize':
            mpl.rc('xtick', labelsize=float(value))
        elif param == 'xtick.direction':
            mpl.rc('xtick', direction=value)
        elif param == 'xtick.major.size':
            mpl.rc('xtick.major', size=float(value))
        elif param == 'xtick.major.width':
            mpl.rc('xtick.major', width=float(value))
        elif param == 'ytick.labelsize':
            mpl.rc('ytick', labelsize=float(value))
        elif param == 'ytick.direction':
            mpl.rc('ytick', direction=value)
        elif param == 'ytick.major.size':
            mpl.rc('ytick.major', size=float(value))
        elif param == 'ytick.major.width':
            mpl.rc('ytick.major', width=float(value))
        elif param == 'cmap':
            plt.rcParams['image.cmap'] = value
        else:
            print('Parameter %s not recognized' % (param))
    def setp(self, param, value):
        """Set plotting parameter

        Args:
            param (str): can be one of 'mathtext.default', 'font.size',
            'axes.labelsize', 'axes.linewidth', 'xtick.labelsize', 'xtick.direction',
            'xtick.major.size', 'xtick.major.width', 'ytick.labelsize', 
            'ytick.direction', 'ytick.major.size', 'ytick.major.width', 'cmap'.
            value: value that parameter will take
        """
        if type(param) == list or type(param) == type(tuple):
            for i, p in enumerate(param):
                self._setp(p, value[i])
        else:
            self._setp(param, value)

class Dataset:
    def __init__(self, filepath, first_file_number, last_file_number,
                 detector='mac', mac_suffix='_reb_0002.xye', 
                 psd_suffix='_summed.xye', log_fname=None,
                 wavelength=0.8527, zpe=0):
        self.filepath = filepath
        self.expt_nums = list(range(first_file_number, last_file_number+1))
        if detector.lower() == 'mac':
            self.detector = 'mac'
        elif detector.lower() == 'psd':
            self.detector = 'psd'
        else:
            print('detector must be "mac" or "psd"')
        self.mac_suffix = mac_suffix
        self.psd_suffix = psd_suffix
        self.beam_offs = []
        self.Tvals = []
        self._assign_log_fname(log_fname)
        self.wavelength = wavelength
        self.zpe = zpe
        
    def get_run_numbers(self):
        """Return array of run numbers"""
        return np.array(self.expt_nums)

    def get_expt_fnames_all(self, scans=True): 
        """Return all experiment file names"""
        if self.detector == 'psd':
            return [self.filepath + str(n) + '-mythen' + self.psd_suffix
                    for n in self.expt_nums]
        elif self.detector == 'mac':
            if scans:
                return [self.filepath + str(n) + '-mac-???' + 
                        self.mac_suffix for n in self.expt_nums]
            else:
                return [self.filepath + str(n) + '-mac-001' + 
                        self.mac_suffix for n in self.expt_nums]

    def get_expt_fnames(self, print_missing=True, scans=True):
        result = []
        #missing = []
        fnames_all = self.get_expt_fnames_all(scans)
        file_list = [self.filepath + fl for fl in 
                     os.listdir(self.filepath)]
        suppress = 0
        if scans:
            for i, f in enumerate(fnames_all):
                matches = []
                for i1, fn in enumerate(file_list):
                    if fnmatch.fnmatch(fn, f):
                        matches.append(fn)
                matches.sort()
                result += matches
                if not matches:
                    if suppress > 0:
                        suppress -= 1
                        continue
                    else:
                        result.append('')
                        if print_missing:
                            print('File %d is missing' % self.expt_nums[i])
                            print(f)
                suppress = len(matches) - 1
        else:
            for i, f in enumerate(fnames_all):
                if f in file_list:
                    result.append(f)
                else:
                    result.append('')
                    #missing.append(self.expt_nums[i])
                    if print_missing:
                        print('File %d is missing' % self.expt_nums[i])
                        print(f)
        return result

    def get_data(self, print_missing=True, scans=True):
        data = []
        first_missing = False
        expt_fnames = self.get_expt_fnames(print_missing, scans)
        bo_indices = [self.expt_nums.index(bo) for bo in self.beam_offs]
        expt_fnames = ['' if i in bo_indices else fname for i, fname in
                       enumerate(expt_fnames)]
        marker = 0
        for i, f in enumerate(expt_fnames):
            if f:
                marker = i
                data.append(pd.read_csv(f, header=None, 
                                        delim_whitespace=True,
                                        names=['x', 'y', 'e']))
            else:
                if len(data):
                    data.append(pd.DataFrame({'x' : data[0]['x'].values, 
                                              'y' : \
                                              np.zeros(data[0].shape[0]),
                                              'e' : \
                                              np.zeros(data[0].shape[0])}))
                else:
                    first_missing=True
        if first_missing:
            data.insert(0, pd.DataFrame({'x' : data[marker - 1]['x'].values, 
                                         'y' : \
                                         np.zeros(data[marker - 1].shape[0]),
                                         'e' : \
                                    np.zeros(data[marker - 1].shape[0])}))
        self.data = data
        return

    def _append_log_fname(self, log_fname):
        if '\\' in log_fname:
            app_lf = log_fname
        elif '/' in log_fname:
            app_lf = log_fname
        else:
            app_lf = self.filepath + log_fname
        return app_lf

    def _assign_log_fname(self, log_fname=None):
        if log_fname:
            if type(log_fname) == type(''):
                log_fname = [log_fname]
            self.log_fname = [self._append_log_fname(lf) for lf in 
                              log_fname]
        else:
            self.log_fname = None

    def get_temps(self, log_fname=None):
        """Fetch temperature vs run number for Datset log file(s)"""
        if log_fname:
            _assign_log_fname(log_fname)
        if not self.log_fname:
            fnames = os.listdir(self.filepath)
            self.log_fname = [self.filepath + fn for fn in fnames if 
                              fn[-3:] == 'log']
        log_data = []
        for lf in self.log_fname:
            log_data.append(pd.read_csv(lf, header=None, 
                                        delim_whitespace=True, 
                                        names=['rn', 'T'], usecols=[0, 1]))
        log_data_rns = np.concatenate([ld.values[:, 0] for ld in 
                                       log_data]).astype(int)
        log_data_Ts = np.concatenate([ld.values[:, 1] for ld in log_data])
        #sort out log_data order
        log_sort_is = np.argsort(log_data_rns)
        self.log_data_rns = log_data_rns[log_sort_is]
        self.log_data_Ts = log_data_Ts[log_sort_is]

    #data_xy() only works if all files are the same length
    #might need future-proofing at some point.
    def data_xy(self, indices=None):
        """Return two arrays of x data and y data
        
        Args:
            indices (list): start and end indices for x data and y data
            respectively [ix0, ix1, iy0, iy1].
        """
        if len(self.data) == 0:
            return
        if not indices:
            indices = [0, len(self.data) - 1, 0, self.data[0].shape[0] - 1]
        ix0, ix1, iy0, iy1 = indices
        data_x = np.zeros((iy1 - iy0 + 1, ix1 - ix0 + 1))
        data_y = np.copy(data_x)
        for i, dset in enumerate(self.data[ix0:ix1 + 1]):
            data_x[:, i] = dset['x'].values[iy0:iy1 + 1]
            data_y[:, i] = dset['y'].values[iy0:iy1 + 1]
        return data_x, data_y
    
    def get_indices(self, rn_range, y_range):
        if type(rn_range) != type(None):
            if rn_range[0] < self.expt_nums[0]:
                rn_range = [rn + self.expt_nums[0] for rn in rn_range]
            indices = [np.searchsorted(self.get_run_numbers(), rn) for rn
                       in rn_range]
        else:
            indices = [0, len(self.data) - 1]
        if type(y_range) != type(None):
            i0, i1 = np.searchsorted(self.data[0].values[:, 1], y_range)
            if i1 >= self.data[0].shape[0]:
                i1 = self.data[0].shape[0] - 1
            indices += [i0, i1]
        else:
            indices += [0, self.data[0].shape[0] - 1]
        return indices

    def get_max_intensity(self, rn_range=None, y_range=None):
        """Return maximum intensity recorded within a certain range
        
        Args:
            rn_range: list of start and end run numbers (if less than
            actual first run number then added on to that).
            y_range: list of start and end y values
        """
        indices = self.get_indices(rn_range, y_range)
        intensity = self.data_xy(indices)[1]
        return intensity.max()

    def get_min_intensity(self, rn_range=None, y_range=None):
        """Return minimum intensity recorded within a certain range
        
        Args:
            rn_range: list of start and end run numbers (if less than
            actual first run number then added on to that).
            y_range: list of start and end y values
        """
        indices = self.get_indices(rn_range, y_range)
        intensity = self.data_xy(indices)[1]
        return intensity.min()

    def get_max_intensities(self, rn_range=None, y_range=None,
                            full_output=True):
        """Return maximum intensities (and y vals) for certain run numbers
        
        Args:
            rn_range: list of start and end run numbers (if less than
            actual first run number then added on to that).
            y_range: list of start and end y values
            full_output (bool): if True, return run_number and y values 
            as well as maximum intensities.
        """
        indices = self.get_indices(rn_range, y_range)
        yvals, intensity = self.data_xy(indices)
        real_is = intensity.sum(axis=0).nonzero()
        max_is = (np.arange(len(real_is[0])), 
                  intensity.argmax(axis=0)[real_is])
        max_ints = intensity.T[real_is][max_is]
        if not full_output:
            return max_ints
        max_ys = yvals.T[real_is][max_is]
        max_rns = self.get_run_numbers()[real_is]
        return max_rns, max_ys, max_ints

    def get_min_intensities(self, rn_range=None, y_range=None,
                            full_output=True):
        """Return minimum intensities (and y vals) for certain run numbers
        
        Args:
            rn_range: list of start and end run numbers (if less than
            actual first run number then added on to that).
            y_range: list of start and end y values
            full_output (bool): if True, return run_number and y values 
            as well as minimum intensities.
        """
        indices = self.get_indices(rn_range, y_range)
        yvals, intensity = self.data_xy(indices)
        real_is = intensity.sum(axis=0).nonzero()
        min_is = (np.arange(len(real_is[0])), 
                  intensity.argmin(axis=0)[real_is])
        min_ints = intensity.T[real_is][min_is]
        if not full_output:
            return min_ints
        min_ys = yvals.T[real_is][min_is]
        min_rns = self.get_run_numbers()[real_is]
        return min_rns, min_ys, min_ints

    def x_range(self):
        if len(self.data) == 0:
            return
        else:
            return [self.data[0].values[0, 0], self.data[0].values[-1, 0]]
    
    def to_xye(self, filepath='', pre_fname='', post_fname='.xye', 
               sep='\t', file_nums=None, tval=None, t=None):
        """Write xye files for all datasets in Dataset
        
        Args:
            filepath (str): file path for destination files (defaults to current
            directory).
            pre_fname (str): precursor string for file names
            post_fname (str): post string for file names (defaults to '.dat')
            sep (str): separator string for columns (defaults to '\t'---tab)
            file_nums (list): list of numbers, defaults to index numbers of 
            Dataset.
            tval: which time/run number to plot
            t: defaults to range(len(data))
        """
        if file_nums:
            if type(file_nums) == type(''):
                file_nums = [file_nums]
            fnames = [filepath + pre_fname + str(f) + post_fname for f in
                      file_nums]
        else:
            fnames = [filepath + pre_fname + str(f) + post_fname for f in 
                      range(len(self.data))]
        if tval:
            if type(t) == type(None):
                t = np.arange(len(self.data))
            ti = np.abs(t - tval).argmin()
            self.data[ti].to_csv(fnames[0], index=False, header=False, 
                                 sep=sep)
            return
        for i in range(len(self.data)):
            self.data[i].to_csv(fnames[i], index=False, header=False, 
                                sep=sep)
        return        
    
    def sum_dsets(self, sum_num, file_range=None, t=None, T=None):
        """Return mean summed datasets (and times) for sum_num interval
        Args:
            sum_num (int): number of patterns to sum over
            file_range: file range (list of two inclusive file numbers) 
            within which to sum data.
            t (arr): time array
            T (arr): temperature array
        Returns:
            result: Dataset of summed/averaged datasets with propogated 
            errors.
            t_result: average time array if t != None or average run number
            if t is None.
            T_result: average temperature array if T != None
        """
        if file_range:
            indices = [self.expt_nums.index(fn) for fn in file_range]
            idxs = indices[:]
        else:
            indices = [0, len(self.expt_nums) - 1]
            idxs = [0, len(self.data) - 1]
        result = []
        for i, dset in enumerate(self.data[idxs[0]:idxs[1] + 1]):
            if i % sum_num == 0:
                if i:
                    if i == len(self.data[idxs[0]:idxs[1] + 1]) - 1:
                        new_y = np.column_stack((new_y, dset['y'].values))
                        new_e = np.column_stack((new_e, dset['e'].values**2))
                        new_e = np.sum(new_e, axis=1)**0.5 / (sum_num + 1)
                    else:
                        new_e = np.sum(new_e, axis=1)**0.5 / sum_num
                    new_dset = pd.DataFrame(np.column_stack(\
                        (dset['x'].values, np.mean(new_y, axis=1), new_e)))
                    new_dset.columns = ['x', 'y', 'e']
                    result.append(new_dset)
                new_y = dset['y'].values
                new_e = dset['e'].values
            else:
                new_y = np.column_stack((new_y, dset['y'].values))
                new_e = np.column_stack((new_e, dset['e'].values**2))
            if i == len(self.data[idxs[0]:idxs[1] + 1]) - 1 and \
               i % sum_num != 0:
                new_e = np.sum(new_e, axis=1)**0.5 / (i % sum_num + 1)
                new_dset = pd.DataFrame(np.column_stack(\
                        (dset['x'].values, np.mean(new_y, axis=1), new_e)))
                new_dset.columns = ['x', 'y', 'e']
                result.append(new_dset)                
        if t is None:
            rns = self.get_run_numbers()[idxs[0]:idxs[1] + 1]
            new_t = np.array([rn - rns[0] for rn in rns])
        else:
            new_t = t[idxs[0]:idxs[1] + 1]
        t_result = self._sum_mean(new_t, idxs, sum_num)
        if T is not None:
            T = T[idxs[0]:idxs[1] + 1]
            T_result = self._sum_mean(T, idxs, sum_num)
        res = Dataset(self.filepath, self.expt_nums[indices[0]], 
                      self.expt_nums[indices[1]], detector=self.detector,
                      mac_suffix=self.mac_suffix, 
                      psd_suffix=self.psd_suffix, log_fname=self.log_fname,
                      wavelength=self.wavelength, zpe=self.zpe)
        res.data = result
        res.expt_nums = t_result
        if type(self.Tvals) == type(np.array([])) and \
                len(self.Tvals):
            new_T = self._sum_mean(self.Tvals[idxs[0]:idxs[1] + 1], 
                    idxs, sum_num)
            res.Tvals = np.array(new_T)
        if T is None and t is None:
            return res
        elif T is not None and t is None:
            return res, T_result
        elif t is not None and T is None:
            return res, t_result
        else:
            return res, t_result, T_result

    def twotheta_to_d(self, wavelength=None, zpe=None, file_range=None):
        """Return Dataset instance with d spacings rather than two theta"""
        if file_range:
            indices = [self.expt_nums.index(fn) for fn in file_range]
            idxs = indices[:]
        else:
            indices = [0, len(self.expt_nums) - 1]
            idxs = [0, len(self.data) - 1]
        if wavelength:
            self.wavelength = wavelength
        if zpe:
            self.zpe = zpe
        result = []
        for i, dset in enumerate(self.data[idxs[0]:idxs[1] + 1]):
            new_x = self.wavelength / (2 * np.sin(((dset['x'].values - \
                                self.zpe) * np.pi)/ 360.))
            new_dset = pd.DataFrame(np.column_stack((new_x, 
                                    dset['y'].values, dset['e'].values)))
            new_dset.columns=['x', 'y', 'e']
            result.append(new_dset)
        res = Dataset(self.filepath, int(self.expt_nums[indices[0]]),
                      int(self.expt_nums[indices[1]]), 
                      detector=self.detector, mac_suffix=self.mac_suffix,
                      psd_suffix=self.psd_suffix, log_fname=self.log_fname,
                      wavelength=self.wavelength, zpe=self.zpe)
        res.data = result
        return res
    
    def plot(self, tval, t=None, xlabel=r'2$\theta$', 
             ylabel='Intensity / Counts', figsize=(10, 7), x_range=None, 
             y_range=None, linecolour=None, labels=None, legend=True,
             legend_loc=0, xclip=True, normalize=False, 
             waterfall_offset_x=0, waterfall_offset_y=0,
             auto_offset_y=False, auto_label=None, auto_label_offsets=0.1,
             no_y_ticks=False):
        """Return a 2D plot of the diffraction data
        
        Args:
            tval: which time/run number to plot (list if more than one)
            t: defaults to range(len(data))
            xlabel: label for x-axis
            ylabel: label for y-axis
            figsize: size of figure (inches by inches)
            x_range (list): x range
            y_range (list): y range
            linecolour (str or list): colour of plotted line(s)
            labels (list): list of labels (if different from tvalues)
            legend (bool): boolean to determine presence of legend
            legend_loc: location of legend
            xclip (bool): whether to clip data according to x_range or not
            normalize (bool): whether to normalize data or not
            waterfall_offset_x: for plotting waterfall plots, x offset
            waterfall_offset_y: for plotting waterfall plots, y offset
            auto_offset_y (bool): whether to automatically offset y or
            not; will override waterfall_offset_y regardless of value.
            auto_label (str): if given value of 'left' or 'right', legend 
            is suppressed and labels are plotted on the relevant side of
            the pattern.
            auto_label_offsets (float or list of floats): fraction(s) of 
            the way up each diffraction pattern that the label will be
            plotted (requires auto_label to be set).
            no_y_ticks (bool): whether to suppress y ticks
        Returns:
            fig: figure instance
            ax: axes instance
        """
        if auto_offset_y:
            waterfall_offset_y = 0
        if t is None:
            t = np.arange(len(self.data))
        if type(tval) == int or type(tval) == float:
            tval = [tval]
        if type(linecolour) == str:
            linecolour = [linecolour]
        if type(labels) == type(None):
            labels = [str(tv) for tv in tval]
        if auto_label:
            legend = False
            if type(auto_label_offsets) == float:
                auto_label_offsets = [auto_label_offsets for l in labels]
        tis = [np.abs(t - tv).argmin() for tv in tval]
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        max_y = 0 #for auto_offset_y
        for i, ti in enumerate(tis):
            if xclip and type(x_range) != type(None):
                xis = [np.abs(self.data[ti]['x'].values - xr).argmin() 
                       for xr in x_range]
                xis.sort()
                data_x = np.copy(self.data[ti]['x'].values[xis[0]:xis[1] \
                                                           + 1])
                data_y = np.copy(self.data[ti]['y'].values[xis[0]:xis[1] \
                                                           + 1])
            else:
                data_x = np.copy(self.data[ti]['x'].values)
                data_y = np.copy(self.data[ti]['y'].values)
            if normalize:
                data_y = (data_y - data_y.min()) / \
                        (data_y - data_y.min()).max()
            data_x += (waterfall_offset_x * i)
            data_y += (waterfall_offset_y * i)
            if auto_offset_y:
                data_y += (max_y - data_y.min())
            if type(linecolour) == type(None):
                line,  = ax.plot(data_x, data_y, label=labels[i])
            else:
                line,  = ax.plot(data_x, data_y, color=linecolour[i], 
                                 label=labels[i])
            max_y = data_y.max()
            if auto_label:
                min_y = data_y.min()
                if type(x_range) == type(None):
                    min_x = data_x.min()
                    max_x = data_x.max()
                else:
                    min_x = x_range[0]
                    max_x = x_range[1]
                ypos = auto_label_offsets[i] * (max_y - min_y) + min_y
                if auto_label == 'left':
                    xpos = 0.02 * (max_x - min_x) + min_x
                else:
                    xpos = 0.98 * (max_x - min_x) + min_x
                lc = line.get_color()
                ax.text(xpos, ypos, labels[i], color=lc, ha=auto_label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if legend:
            ax.legend(loc=legend_loc)
        if type(x_range) != type(None):
            ax.set_xlim(x_range[0], x_range[1])
        if type(y_range) != type(None):
            ax.set_ylim(y_range[0], y_range[1])
        if no_y_ticks:
            ax.set_yticks([])
            ax.set_yticklabels([])
        fig.tight_layout() 
        return fig, ax

    def get_deltad_over_d(self, tval, dvals, t=None):
        """Return list of delta d over d arrays for tvals and dvals

        Args:
            tval: t value (i.e. run number)
            dvals: list of d values to act as denominator
            t: array of time values
        Returns:
            result: list of arrays of delta d over d values
        """
        if t is None:
            t = np.arange(len(self.data))
        if type(dvals) == int or type(dvals) == float:
            dvals = [dvals]
        ti = np.searchsorted(t, tval)
        result = [(self.data[ti]['x'].values - dval) / dval for dval in 
                  dvals]
        return result

    def plot_deltad_over_d(self, tval, dvals, t=None, 
                           xlabel=r'$\frac{\Delta d}{d}$',
                           ylabel='Normalized Intensity', figsize=(10, 7), 
                           x_range=None, y_range=None, linecolour=None, 
                           labels=None, legend=True, legend_loc=0,
                           bgd_pts=8):
        """Return plot of delta d over d versus intensity

        Args:
            tval: t value (i.e. run number)
            dvals: list of d values to act as denominator
            t: array of time values
            xlabel: label for x-axis
            ylabel: label for y-axis
            figsize: size of figure (inches by inches)
            x_range (list): x range
            y_range (list): y range
            linecolour (str or list): colour of plotted line(s)
            labels (list): list of labels (if different from tvalues)
            legend (bool): boolean to determine presence of legend
            legend_loc: location of legend
            bgd_pts (int): number of points on either side of peak to 
            use to take a linear background for normalizing intensity
        Returns:
            fig: figure instance
            ax: axes instance
        """
        if t is None:
            t = np.arange(len(self.data))
        ti = np.searchsorted(t, tval)
        xvals = self.get_deltad_over_d(tval, dvals, t=t)
        if x_range is None:
            x_range = [xvals[-1].min(), xvals[0].max()]
        if labels is None:
            labels = [str(n) for n in range(len(xvals))]
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        for i, xval in enumerate(xvals):
            xis = [np.abs(xval - xr).argmin() for xr in x_range]
            xis.sort()
            data_x = xval[xis[0]:xis[1] + 1]
            data_y = self.data[ti]['y'].values[xis[0]:xis[1] + 1]
            m, c = np.polyfit(np.concatenate((data_x[:bgd_pts], 
                                              data_x[-bgd_pts:])),
                              np.concatenate((data_y[:bgd_pts], 
                                              data_y[-bgd_pts:])), 1)
            data_y = data_y - (data_x * m + c)
            data_y = data_y / data_y.max()
            if type(linecolour) == type(None):
                ax.plot(data_x, data_y, label=labels[i])
            else:
                ax.plot(data_x, data_y, color=linecolour[i], 
                        label=labels[i])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if legend:
            ax.legend(loc=legend_loc)
        fig.tight_layout() 
        return fig, ax

    def plot_reflected_peak(self, tval, t=None, xlabel=r'2$\theta$', 
                            ylabel='Intensity / Counts', figsize=(10, 7),
                            x_range=None, y_range=None, linecolour=None, 
                            labels=None, legend=True, legend_loc=0,
                            take_bgd=True, bgd_pts=8, centrepoint=None, 
                            centreline=True):
        """Return plot of delta d over d versus intensity

        Args:
            tval: t value (i.e. run number)
            t: array of time values
            xlabel: label for x-axis
            ylabel: label for y-axis
            figsize: size of figure (inches by inches)
            x_range (list): x range
            y_range (list): y range
            linecolour (str or list): colour of plotted line(s)
            labels (list): list of labels (if different from tvalues)
            legend (bool): boolean to determine presence of legend
            legend_loc: location of legend
            take_bgd (bool): whether to remove a linear background or not
            bgd_pts (int): number of points on either side of peak to 
            use to take a linear background for normalizing intensity
            centrepoint: point about which to reflect
            centreline (bool): whether to plot line of reflection or not
        Returns:
            fig: figure instance
            ax: axes instance
        """
        if t is None:
            t = np.arange(len(self.data))
        ti = np.searchsorted(t, tval)
        if x_range is None:
            x_range = [self.data[0]['x'].iloc[0],
                       self.data[0]['x'].iloc[-1]]
        xis = [np.abs(self.data[ti]['x'].values - xr).argmin() for xr in 
               x_range]
        if labels is None:
            labels = ['original', 'reflected']
        data_x = self.data[ti]['x'].values
        data_y = self.data[ti]['y'].values
        if take_bgd:
            xvals = np.concatenate((data_x[xis[0]:xis[1] + 1][:bgd_pts],
                                    data_x[xis[0]:xis[1] + 1][-bgd_pts:]))
            yvals = np.concatenate((data_y[xis[0]:xis[1] + 1][:bgd_pts],
                                    data_y[xis[0]:xis[1] + 1][-bgd_pts:]))
            m, c = np.polyfit(xvals, yvals, 1)
            data_y = data_y - (data_x * m + c)
        if centrepoint is None:
            cum_y = np.cumsum(data_y[xis[0]:xis[1] + 1])
            centre_i = np.searchsorted(cum_y, cum_y[-1] / 2.) + xis[0]
        else:
            centre_i = np.searchsorted(data_x, centrepoint)
        refl_x = 2 * data_x[centre_i] - data_x
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        if linecolour is None:
            ax.plot(data_x, data_y, label=labels[0])
            ax.plot(refl_x, data_y, label=labels[1])
        else:
            ax.plot(data_x, data_y, label=labels[0], color=linecolour[0])
            ax.plot(refl_x, data_y, label=labels[1], color=linecolour[1])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if legend:
            ax.legend(loc=legend_loc)
        if type(x_range) != type(None):
            ax.set_xlim(x_range[0], x_range[1])
        if type(y_range) != type(None):
            ax.set_ylim(y_range[0], y_range[1])
        if centreline:
            ax.axvline(data_x[centre_i], linestyle='dashed', color='k') 
        fig.tight_layout() 
        return fig, ax
       
    def contour_plot(self, t=None, xlabel='Run number', 
                     ylabel=r'2$\theta$', zlabel='Intensity / Counts', 
                     colour_num=20, figsize=(10, 7), x_range=None, 
                     y_range=None, z_range=None, xyflip=False, yflip=False,
                     zscale=None, log_zlabel='log(Intensity / Counts)', 
                     sqrt_zlabel='$\sqrt{Intensity / Counts}$'):
        """Return a contour plot of the data
        
        Args:
            t: array of time/run number values, can be 1D or 2D
            xlabel: label for x axis
            ylabel: label for y axis
            zlabel: label for colour bar
            colour_num: number of colours in colour bar
            figsize: size of the figure (inches by inches)
            x_range (list): x range
            y_range (list): y range
            z_range (list): z range
            xyflip (bool): determines which value on x-axis (default 
            False=time)
            yflip (bool): determines whether to flip the y axis
            zscale: 'log' for log scaling and 'sqrt' for square root
            log_zlabel (str): Title of colourbar when zscale='log'
            sqrt_zlabel (str): Title of colourbar when zscale='sqrt'
        Returns:
            fig: figure instance
            ax: axes instance
        """
        #26/01/16 rewrite
        #data_y, data_z = self.data_xy() #data_y is 2theta, data_z is intensity
        if type(t) == type(None):
            rns = self.get_run_numbers()
            t = np.array([rn - rns[0] for rn in rns])
        if t.ndim == 1:
            t = np.meshgrid(t, np.arange(self.data[0].shape[0]))[0]
        if x_range:
            ix0, ix1 = [np.abs(t[0, :] - val).argmin() for val in x_range]
            t = t[:, ix0:ix1 + 1]
        else:
            ix0, ix1 = 0, len(self.data) - 1
        if y_range:
            iy0, iy1 = [np.abs(self.data[0].values[:, 0] - val).argmin() for
                        val in y_range]
            t = t[iy0:iy1 + 1, :]
        else:
            iy0, iy1 = 0, self.data[0].shape[0] - 1
        data_y, data_z = self.data_xy(indices=[ix0, ix1, iy0, iy1])
        if z_range:
            data_z = np.clip(data_z, z_range[0], z_range[1])
        if zscale == 'log':
            data_z = np.log10(data_z)
        elif zscale == 'sqrt':
            data_z = np.sqrt(data_z)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        if xyflip:
            plt.contourf(data_y, t, data_z, colour_num)
            ax.set_xlabel(ylabel)
            ax.set_ylabel(xlabel)
        else:
            plt.contourf(t, data_y, data_z, colour_num)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
        if yflip:
            ax.invert_yaxis()
        plt.tick_params(which='both', top=False, right=False)
        cbar = plt.colorbar()
        if zscale == 'log':
            cbar.set_label(log_zlabel, rotation=270, labelpad=20)
        elif zscale == 'sqrt':
            cbar.set_label(sqrt_zlabel, rotation=270, labelpad=30)
        else:
            cbar.set_label(zlabel, rotation=270, labelpad=20)
        fig.tight_layout()
        return fig, ax
        
    def contour_temp(self, T=None, t=None, xlabel='Run number', 
                     ylabel=r'2$\theta$', ylabel2=u'Temperature / \u00B0C',
                     zlabel='Intensity / Counts', colour_num=20, 
                     figsize=(10, 7), x_range=None, y_range=None, 
                     z_range=None, yflip=False, Tcolour='g', 
                     height_ratios=[1, 2], zscale=None, 
                     log_zlabel='log(Intensity / Counts)',
                     sqrt_zlabel = '$\sqrt{Intensity / Counts}$'):
        """Return a contour plot of the data
        
        Args:
            T: temperature array
            t: array of time/run number values, can be 1D or 2D
            xlabel: label for x axis
            ylabel: label for y axis
            ylabel: label for temperature plot
            zlabel: label for colour bar
            colour_num: number of colours in colour bar
            figsize: size of the figure (inches by inches)
            x_range (list): x range
            y_range (list): y range
            z_range (list): z range
            yflip (bool): determines whether to flip the y axis
            Tcolour (str): determines colour of temperature plot
            height_ratios: ratios of heights of subplots
            zscale: can be 'log' or 'sqrt'
            log_zlabel (str): Title of colourbar when zscale='log'
            sqrt_zlabel (str): Title of colourbar when zscale='sqrt'

        Returns:
            fig: figure instance
            ax1: Temperature line plot axes
            ax2: contour plot axes
        """
        data_y, data_z = self.data_xy() #data_y is 2theta, data_z is I(2th)
        if type(t) == type(None):
            rns = self.get_run_numbers()
            t = np.array([rn - rns[0] for rn in rns])
        if type(T) == type(None):
            T_t = np.array([rn - self.get_run_numbers()[0] for rn in 
                            self.log_data_rns])
            T = self.log_data_Ts
        else:
            if t.ndim == 1:
                T_t = t
            else:
                T_t = t[0]
        if t.ndim == 1:
            t = np.meshgrid(t, np.arange(data_y.shape[0]))[0]
        if x_range:
            i0, i1 = [np.abs(t[0, :] - val).argmin() for val in x_range]
            t = t[:, i0:i1 + 1]
            data_y = data_y[:, i0:i1 + 1]
            data_z = data_z[:, i0:i1 + 1]
            i2, i3 = [np.searchsorted(T_t, val) for val in x_range]
            T_t = T_t[i2:i3 + 1]
        if y_range:
            i0, i1 = [np.abs(data_y[:, 0] - val).argmin() for val in 
                      y_range]
            t = t[i0:i1 + 1, :]
            data_y = data_y[i0:i1 + 1, :]
            data_z = data_z[i0:i1 + 1, :]
        if z_range:
            data_z = np.clip(data_z, z_range[0], z_range[1])
        if zscale == 'log':
            data_z = np.log10(data_z)
        elif zscale == 'sqrt':
            data_z = np.sqrt(data_z)
        fig = plt.figure(figsize=figsize)
        gs = mpl.gridspec.GridSpec(2, 1, height_ratios=height_ratios)
        ax2 = fig.add_subplot(gs[1, 0])
        ax1 = fig.add_subplot(gs[0, 0], sharex=ax2)
        cont = ax2.contourf(t, data_y, data_z, colour_num)
        if yflip:
            ax2.invert_yaxis()
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel(ylabel)
        ax2.tick_params(which='both', top=False, right=False, 
                        direction='out')
        ax2.set_xlim(t[0, 0], t[0, -1])
        ax1.plot(T_t, T, color=Tcolour)
        ax1.set_ylabel(ylabel2)
        ax1.tick_params(which='both', top=False, right=False, 
                        direction='out')
        plt.setp(ax1.get_xticklabels(), visible=False)
        axins = inset_axes(ax2, width='5%', height='100%', loc=6,
                           bbox_to_anchor=(1.05, 0., 1, 1), borderpad=0,
                           bbox_transform=ax2.transAxes)
        cbar = plt.colorbar(cont, cax=axins)
        if zscale == 'log':
            cbar.set_label(log_zlabel, rotation=270, labelpad=20)
        elif zscale == 'sqrt':
            cbar.set_label(sqrt_zlabel, rotation=270, labelpad=30)
        else:
            cbar.set_label(zlabel, rotation=270, labelpad=20)
        ax2.set_xlim(t[0, 0], t[0, -1])
        fig.tight_layout(rect=(0, 0, 0.85, 1))
        return fig, ax1, ax2
    
    def contour_mult(self, T=None, t=None, xlabel='Run number', 
                     ylabel=u'd / \u00C5', ylabel2=u'Temperature / \u00B0C',
                     zlabel='Intensity / Counts', colour_num=20, 
                     figsize=(10, 7), x_range=None, y_range=None, 
                     z_range=None, xyflip=False, Tcolour='g', 
                     height_ratios=None, width_ratios=None, zscale=None, 
                     log_zlabel='log(Intensity / Counts)',
                     sqrt_zlabel = '$\sqrt{Intensity / Counts}$', grid=None, 
                     sharey=True, shareT=True, colourbar=False):
        """Return a contour plot of the data
        
        Args:
            T: temperature array
            t: array of time/run number values, can be 1D or 2D
            xlabel: label for x axis
            ylabel: label for y axis
            ylabel: label for temperature plot
            zlabel: label for colour bar
            colour_num: number of colours in colour bar
            figsize: size of the figure (inches by inches)
            x_range (list or list of lists): x range(s)
            y_range (list or list of lists): y range(s)
            z_range (list or list of lists): z range(s)
            height_ratios: ratios of heights of subplots
            zscale: can be 'log' or 'sqrt' or list of those (or None)
            grid: if None then defined by len(x_range) and len(y_range) 
            otherwise is a list of lists of x,y axes to be populated.
            sharey (bool): assume that y-axes are shared intra rows, 
            otherwise separate ticks for each one.
            shareT (bool): assume that Temp axes are shared intra rows, 
            otherwise separate ticks for each one.
            colourbar (bool): determines presence of colour bar(s).

        Returns:
            fig: figure instance
            axes: list of axes instances
        """
        data_y, data_z = self.data_xy() #data_y is 2theta, data_z is I(2th)
        #work out grid space needed
        if grid:
            gs_row = [g[0] for g in grid].max()
            gs_col = [g[1] for g in grid].max()
        else:
            if type(x_range) == type(None):
                gs_col = 1
            else:
                if type(x_range[0]) == list or type(x_range[0]) == type(None):
                    gs_col = len(x_range)
                else:
                    gs_col = 1
            if type(y_range) == type(None):
                gs_row = 1
            else:
                if type(y_range[0]) == list or type(y_range[0]) == type(None):
                    gs_row = len(y_range)
                    if not sharey:
                        gs_row /= gs_col
                else:
                    gs_row = 1
        #make t the same shape as data_y and data_z
        if type(t) == type(None):
            t = np.meshgrid(np.arange(data_y.shape[1]), 
                    np.arange(data_y.shape[0]))[0]
        elif t.ndim == 1:
            t = np.meshgrid(t, np.arange(data_y.shape[0]))[0]
        #separate out datasets for each contour plot and T for each x_range
        t_indices = []
        y_indices = []
        if type(x_range) == type(None): #convert x_range into list of lists
            x_range = [[t[0, 0], t[0, -1]]]
        elif type(x_range) == list and type(x_range[0]) != list and \
        type(x_range[0]) != type(None):
            x_range = [x_range]
        for xr in x_range:
            if xr:
                i0, i1 = [np.abs(t[0, :] - val).argmin() for val in xr]
            else:
                i0, i1 = [0, len(t[0, :]) - 1]
            t_indices.append([i0, i1])
        if type(y_range) == type(None): #convert y_range into list of lists
            y_range = [[data_y[0, 0], data_y[-1, 0]]]
        elif type(y_range) == list and type(y_range[0]) != list and \
        type(y_range[0]) != type(None):
            y_range = [y_range]
        for yr in y_range:
            if yr:
                i0, i1 = [np.abs(data_y[:, 0] - val).argmin() for val in yr]
            else:
                i0, i1 = [0, len(data_y[:, 0]) - 1]
            y_indices.append([i0, i1])
        #do arrays here, one array per dimension per grid spot
        if not grid:
            grid = [[i, j] for i in range(gs_row) for j in range(gs_col)]
        t_arrs = []
        y_arrs = []
        z_arrs = []
        for i, g in enumerate(grid):
            ti = t_indices[g[1]]
            if sharey:
                yi = y_indices[g[0]]
            else:
                yi = y_indices[i]
            t_arrs.append(t[yi[0]:yi[1] + 1, ti[0]:ti[1] + 1])
            y_arrs.append(data_y[yi[0]:yi[1] + 1, ti[0]:ti[1] + 1])
            z_arrs.append(data_z[yi[0]:yi[1] + 1, ti[0]:ti[1] + 1])
        #now deal with z_range issues
        if type(z_range) == list and type(z_range[0]) != list and \
        type(z_range[0]) != type(None):
            z_range = [z_range]
        if z_range and len(z_range) == len(z_arrs):
            z_arrs = [np.clip(z_arr, z_range[i][0], z_range[i][1]) for i, z_arr
                      in enumerate(z_arrs)]
        elif z_range and len(z_range) != len(z_arrs):
            z_arrs = [np.clip(z_arr, z_range[0][0], z_range[0][1]) for z_arr
                      in z_arrs]
        if zscale == 'log':
            z_arrs = [np.log10(z_arr) for z_arr in z_arrs]
        elif zscale == 'sqrt':
            z_arrs = [np.sqrt(z_arr) for z_arr in z_arrs]
        #deal with T 
        if type(T) != type(None):
            gs_row += 1
            grid = [[g[0] + 1, g[1]] for g in grid]
            T_arrs = []
            for ti in t_indices:
                T_arrs.append(T[ti[0]:ti[1] + 1])
        #now set up the figure
        fig = plt.figure(figsize=figsize)
        gs = mpl.gridspec.GridSpec(gs_row, gs_col, height_ratios=height_ratios,
                                   width_ratios=width_ratios)
        if colourbar:
            gs.update(wspace=0.5, right=0.85)
        if type(T) != type(None):
            T_axes = []
            for i, T_arr in enumerate(T_arrs):
                if not i:
                    ax = fig.add_subplot(gs[0, 0])
                    ax.set_ylabel(ylabel2)
                if i and shareT:
                    ax = fig.add_subplot(gs[0, i], sharey=T_axes[0])
                elif i and not shareT:
                    ax = fig.add_subplot(gs[0, i])
                ax.plot(t_arrs[i][0, :], T_arr, color=Tcolour)
                if i in [g[1] for g in grid]:
                    plt.setp(ax.get_xticklabels(), visible=False)
                if i and shareT:
                    plt.setp(ax.get_yticklabels(), visible=False)
                ax.tick_params(which='both', top=False, right=False, direction='out')
                T_axes.append(ax)
        #now plot everything else
        axes = []
        fr_indices, lc_indices = [0] * gs_row, [0] * gs_col #used for establishing axis labels
        fr_vals, lc_vals = [gs_col] * gs_row, [-1] * gs_col
        if type(T) != type(None):
            fr_indices[0] = None
        rows, cols = ['x'] * gs_row, ['x'] * gs_col #used for sharex and sharey order proofing
        for i, g in enumerate(grid):
            #amend first/last in row and last in column
            if g[1] < fr_vals[g[0]]:
                fr_vals[g[0]] = g[1]
                fr_indices[g[0]] = i
            if g[0] > lc_vals[g[1]]:
                lc_vals[g[1]] = g[0]
                lc_indices[g[1]] = i
        for i, g in enumerate(grid):
            if type(T) == type(None):
                if rows[g[0]] == 'x' and cols[g[1]] == 'x':
                    ax = fig.add_subplot(gs[g[0], g[1]])
                    rows[g[0]] = i
                    cols[g[1]] = i
                elif rows[g[0]] != 'x' and cols[g[1]] == 'x':
                    if sharey:
                        ax = fig.add_subplot(gs[g[0], g[1]], sharey=axes[rows[g[0]]])
                    else:
                        ax = fig.add_subplot(gs[g[0], g[1]])
                    cols[g[1]] = i
                elif rows[g[0]] == 'x' and cols[g[1]] != 'x':
                    ax = fig.add_subplot(gs[g[0], g[1]], sharex=axes[cols[g[1]]])
                    rows[g[1]] = i
                else:
                    if sharey:
                        ax = fig.add_subplot(gs[g[0], g[1]], sharex=axes[cols[g[1]]],
                                             sharey=axes[rows[g[0]]])
                    else:
                        ax = fig.add_subplot(gs[g[0], g[1]], sharex=axes[cols[g[1]]])
            else:
                if rows[g[0]] == 'x':
                    ax = fig.add_subplot(gs[g[0], g[1]], sharex=T_axes[g[1]])
                    rows[g[0]] = i
                elif rows[g[0]] != 'x' and sharey:
                    ax = fig.add_subplot(gs[g[0], g[1]], sharex=T_axes[g[1]],
                                         sharey=axes[rows[g[0]]])
                else:
                    ax = fig.add_subplot(gs[g[0], g[1]], sharex=T_axes[g[1]])
            cont = ax.contourf(t_arrs[i], y_arrs[i], z_arrs[i], colour_num)
            ax.tick_params(which='both', top=False, right=False, direction='out')
            ax.set_xlim(t_arrs[i][0, 0], t_arrs[i][0, -1])
            if i in fr_indices:
                ax.set_ylabel(ylabel)
            else:
                if sharey:
                    plt.setp(ax.get_yticklabels(), visible=False)
            if i in lc_indices:
                ax.set_xlabel(xlabel)
            else:
                plt.setp(ax.get_xticklabels(), visible=False)
            if colourbar:
                axins = inset_axes(ax, width='5%', height='100%', loc=6,
                                   bbox_to_anchor=(1.05, 0., 1, 1), borderpad=0,
                                   bbox_transform=ax.transAxes)
                cbar = plt.colorbar(cont, cax=axins)
                if zscale == 'log':
                    cbar.set_label(log_zlabel, rotation=270, labelpad=20)
                elif zscale == 'sqrt':
                    cbar.set_label(sqrt_zlabel, rotation=270, labelpad=30)
                else:
                    cbar.set_label(zlabel, rotation=270, labelpad=20)
            axes.append(ax)
        return fig, axes
    
    def plot_animate(self, t_range, t=None, xlabel=u'd / \u00C5',
                     ylabel='Intensity / Counts', figsize=(10, 7), x_range=None,
                     y_range=None, linecolour='g', interval=500, t_text='Run',
                     T=None, T_text='Temp', save_fname=None):
        """Return a 2D animated plot of the diffraction data
        
        Args:
            tvals: which times/run numbers to plot between
            t: defaults to range(len(data))
            xlabel: label for x-axis
            ylabel: label for y-axis
            figsize: size of figure (inches by inches)
            x_range (list): x range
            y_range (list): y range
            linecolour (str): colour of plotted line
            interval: interval in ms between each plot"""
        if type(t) == type(None):
            t = np.arange(len(self.data))
        ti_start, ti_end = [np.abs(t - tval).argmin() for tval in t_range]
        t_indices = list(range(ti_start, ti_end + 1, 1))
        frames = len(t_indices)
        if type(x_range) == type(None):
            x_range = [np.array([dset['x'].min() for dset in self.data]).min(),
                       np.array([dset['x'].max() for dset in self.data]).max()]
        if type(y_range) == type(None):
            y_range = [np.array([dset['y'].min() for dset in self.data]).min(),
                       np.array([dset['y'].max() for dset in self.data]).max()]        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        ax.set_xlim(x_range[0], x_range[1])
        ax.set_ylim(y_range[0], y_range[1])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        line, = ax.plot([], [], color=linecolour)
        run_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
        if type(T) != type(None):
            temp_text = ax.text(0.02, 0.9, '', transform=ax.transAxes)
        def init():
            line.set_data([], [])
            run_text.set_text('')
            if type(T) != type(None):
                temp_text.set_text('')
                return line, run_text, temp_text
            return line, run_text
        def animate(i):
            x = self.data[t_indices[i]]['x'].values
            y = self.data[t_indices[i]]['y'].values
            line.set_data(x, y)
            run_text.set_text('%s = %.0f' % (t_text, t[t_indices[i]]))
            if type(T) != type(None):
                temp_text.set_text('%s = %.0f' % (T_text, T[t_indices[i]]))
                return line, run_text, temp_text
            return line, run_text
        self.anim = animation.FuncAnimation(fig, animate, init_func=init, 
                                            frames=frames, interval=interval, 
                                            blit=True)
        if save_fname:
            self.anim.save(save_fname, fps=10)
        plt.show()
    
    def contour_animate(self, T=None, t=None, xlabel='Run number', 
                        ylabel=u'd / \u00C5', 
                        ylabel2=u'Temperature / \u00B0C', 
                        zlabel='Intensity / Counts',
                        colour_num=20, figsize=(10, 10), x_range=None, 
                        y_range=None, z_range=None, xyflip=False, 
                        Tcolour='g', height_ratios=None, zscale=None, 
                        log_zlabel='log(Intensity / Counts)',
                        sqrt_zlabel = '$\sqrt{Intensity\ /\ Counts}$', 
                        linecolour='g', vline_colour='w', Tpt_colour='r', 
                        Tpt_size=10, interval=200, save_fname=None):
        """Return a contour plot of the data
        
        Args:
            T: temperature array
            t: array of time/run number values, can be 1D or 2D
            xlabel: label for x axis
            ylabel: label for y axis
            ylabel2: label for temperature plot
            zlabel: label for colour bar
            colour_num: number of colours in colour bar
            figsize: size of the figure (inches by inches)
            x_range (list): x range
            y_range (list): y range
            z_range (list): z range
            height_ratios: ratios of heights of subplots
            zscale: can be 'log' or 'sqrt'
        """
        data_y, data_z = self.data_xy() #data_y is 2theta, data_z is intensity
        if type(t) == type(None):
            t = np.meshgrid(np.arange(data_y.shape[1]), 
                    np.arange(data_y.shape[0]))[0]
        elif t.ndim == 1:
            t = np.meshgrid(t, np.arange(data_y.shape[0]))[0]
        if x_range:
            i0, i1 = [np.abs(t[0, :] - val).argmin() for val in x_range]
            t = t[:, i0:i1 + 1]
            data_y = data_y[:, i0:i1 + 1]
            data_z = data_z[:, i0:i1 + 1]
            if type(T) != type(None):
                T = T[:, i0:i1]
        if y_range:
            i0, i1 = [np.abs(data_y[:, 0] - val).argmin() for val in y_range]
            t = t[i0:i1 + 1, :]
            data_y = data_y[i0:i1 + 1, :]
            data_z = data_z[i0:i1 + 1, :]
        else:
            y_range = [data_y.min(), data_y.max()]
        if z_range:
            data_z = np.clip(data_z, z_range[0], z_range[1])
        else:
            z_range = [data_z.min(), data_z.max()]
        if zscale == 'log':
            data_z = np.log10(data_z)
            z_range = [np.log10(zval) for zval in z_range]
        elif zscale == 'sqrt':
            data_z = np.sqrt(data_z)
            z_range = [np.sqrt(zval) for zval in z_range]
        fig = plt.figure(figsize=figsize)
        if type(T) == type(None):
            gs = mpl.gridspec.GridSpec(2, 1, height_ratios=height_ratios)
            ax_cont = fig.add_subplot(gs[0, 0])
            ax_anim = fig.add_subplot(gs[1, 0])
        else:
            gs = mpl.gridspec.GridSpec(3, 1, height_ratios=height_ratios)
            ax_T = fig.add_subplot(gs[0, 0])
            ax_cont = fig.add_subplot(gs[1, 0], sharex=ax_T)
            ax_anim = fig.add_subplot(gs[2, 0])
            ax_T.plot(t[0, :], T, color=Tcolour)
            ax_T.set_ylabel(ylabel2)
            ax_T.tick_params(which='both', top=False, right=False, direction='out')
            plt.setp(ax_T.get_xticklabels(), visible=False)
        cont = ax_cont.contourf(t, data_y, data_z, colour_num)
        ax_cont.set_xlabel(xlabel)
        ax_cont.set_ylabel(ylabel)
        ax_cont.tick_params(which='both', top=False, right=False, direction='out')
        ax_cont.set_xlim(t[0, 0], t[0, -1])
        axins = inset_axes(ax_cont, width='5%', height='100%', loc=6,
                           bbox_to_anchor=(1.05, 0., 1, 1), borderpad=0,
                           bbox_transform=ax_cont.transAxes)
        cbar = plt.colorbar(cont, cax=axins)
        if zscale == 'log':
            cbar.set_label(log_zlabel, rotation=270, labelpad=20)
            ax_anim.set_ylabel(log_zlabel)
        elif zscale == 'sqrt':
            cbar.set_label(sqrt_zlabel, rotation=270, labelpad=30)
            ax_anim.set_ylabel(sqrt_zlabel)
        else:
            cbar.set_label(zlabel, rotation=270, labelpad=20)
            ax_anim.set_ylabel(zlabel)
        ax_cont.set_xlim(t[0, 0], t[0, -1])
        ax_anim.set_xlabel(ylabel)
        ax_anim.set_xlim(y_range[0], y_range[1])
        ax_anim.set_ylim(z_range[0], z_range[1])
        ax_anim.tick_params(which='both', top=False, right=False, direction='out')
        plt.tight_layout(rect=(0.02, 0, 0.85, 1))
        line, = ax_anim.plot([], [], color=linecolour)
        vline = ax_cont.axvline(0, c=vline_colour)
        if type(T) != type(None):
            T_pt, = ax_T.plot([], [], color=Tpt_colour, marker='o', ms=Tpt_size)
        frames = len(t[0, :])
        def init():
            line.set_data([], [])
            vline.set_data([], [])
            if type(T) != type(None):
                T_pt.set_data([], [])
                return line, vline, T_pt
            return line, vline
        def animate(i):
            x = data_y[:, i] #2theta
            y = data_z[:, i] #Intensity
            line.set_data(x, y)
            vline.set_data(t[0, i], [0, 1])
            if type(T) != type(None):
                T_pt.set_data(t[0, i], T[i])
                return line, vline, T_pt
            return line, vline
        self.anim = animation.FuncAnimation(fig, animate, init_func=init, 
                                            frames=frames, interval=interval, 
                                            blit=True)
        if save_fname:
            self.anim.save(save_fname, fps=10)
        plt.show()    
        
    def _sum_mean(self, v, idxs, sum_num):
        """For use within sum_dsets (takes mean of v)"""
        result = []
        for i in range(len(self.data[idxs[0]:idxs[1] + 1])):
            if i % sum_num == 0:
                if i:
                    if i == len(self.data[idxs[0]:idxs[1] + 1]) - 1:
                        v_sum = np.concatenate((v_sum, np.array([v[i]])))
                    result.append(np.mean(v_sum))
                v_sum = np.array([v[i]])
            else:
                v_sum = np.concatenate((v_sum, np.array([v[i]])))
            if i == len(self.data[idxs[0]:idxs[1] + 1]) - 1 and \
               i % sum_num != 0:
                result.append(np.mean(v_sum))
        return result
    
    def _print_shapes(self):
        """Used for debugging purposes. Prints shapes of all datasets"""
        for dset in self.data:
            print(dset.shape)
    def _get_shapes(self):
        """Used for debugging purposes. Returns shapes of all datasets"""
        return [dset.shape for dset in self.data]
