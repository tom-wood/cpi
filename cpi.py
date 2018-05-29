import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd #used mainly for the pd.read_csv function (very fast)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes #for colour bars
from matplotlib import animation #for animation
import os #for finding out which files are in a directory
import re
#from sys import platform
#from IPython import get_ipython
#ipy = get_ipython()
#if platform == 'linux2':
#    ipy.magic("matplotlib qt5")
#else:
#    ipy.magic("matplotlib qt")

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
            print 'Parameter %s not recognized' % (param)
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
                 beamline, beam_min=100., bank_num=4, tof=False):
        self.filepath = filepath
        self.expt_nums = range(first_file_number, last_file_number + 1)
        if beamline == 'POLARIS':
            self.beamline = 'Polaris'
        elif beamline == 'GEM':
            self.beamline == 'Gem'
        else:
            self.beamline = beamline
        self.beam_min = beam_min
        self.bank_num = bank_num
        self.tof = tof
        self.scan_times = []
        self.lstarts = []
        self.lends = []
        self.av_bcs = []
        self.beam_offs = []
        self.beam_offs2 = []
        self.beam_offs3 = []
        self.T_vals = []

    def get_from_log(self, strings):
        """Return all time and values for string within log files

        Args:
            strings (str or list): string(s) to be looked for in log file

        Returns:
            result: list of arrays with times on one column and string
            values on the other.
        """
        if self.beamline == 'Polaris':
            fname_pre = 'POL'
            lflocation = r'\\isis\inst$\ndxpolaris\Instrument\data'
        elif self.beamline == 'Gem':
            fname_pre = 'GEM'
            lflocation = r'\\isis\inst$\ndxgem\Instrument\data'
        else:
            fname_pre = ''
            print('Recognised beamlines are "Polaris" or "Gem"')
        if self.filepath:
            fnames = os.listdir(self.filepath)
            log_files = [fname for fname in fnames if '.log' in fname]
            if len(log_files) > 0:
                if log_files[0][3] == 'A':
                    fname_pre += 'ARIS00' #hack for new Polaris files
                    print('Modified Polaris log filenames')
        else:
            log_files = []
        log_fnames = [self.filepath + fname_pre + str(n) + '.log' for n in
                      self.expt_nums]
        if type(strings) == str:
            strings = [strings]
        first_log = True
        s_tvals = [[] for s in strings]
        s_vals = [[] for s in strings]
        for i1, lf in enumerate(log_fnames):
            if re.split(r'\\|/', lf)[-1] not in log_files:
                print("%s doesn't exist. Try looking in %s for files." \
                        % (lf, lflocation))
                return
            log_data = pd.read_csv(lf, header=None, delim_whitespace=True,
                                   names=['Time', 'String', 'Value'])
            if first_log:
                lstart = np.datetime64(log_data.iloc[0, 0])
                first_log = False
            for i_s, s in enumerate(strings):
                s_is = np.where(log_data.iloc[:, 1].values == s)
                s_tvals[i_s].append((log_data.values[:, 0][s_is]))
                s_vals[i_s].append(log_data.values[:, 2][s_is])
        #if True:
        #    return s_tvals, s_vals
        s_tvals = [np.concatenate([np.array([(np.datetime64(s0) - lstart)/\
                     np.timedelta64(3600, 's') for s0 in s1]) 
                                   for s1 in s2]) for s2 in s_tvals]
        s_vals = [np.concatenate([np.array(s0, dtype='float64') for s0 in
                                  sv]) for sv in s_vals]
        result = [np.column_stack((st, s_vals[i])) for i, st in 
                  enumerate(s_tvals)]
        return result

    def get_scan_times(self, Tstring=None, beam_off_time=120,
                       dataset=None): 
        """Assign log starts, ends, T and beam current attributes
        
        Args:
            Tstring: if temperature wanted from the log files; set to True
            to use default (only tested on Polaris) or to string if 
            bespoke needed.
            beam_off_time: time after which to consider the beam as off
            in seconds (should be longer than single scan time).
            dataset: Dataset from which to get values (to save on memory
            and computational time).
        """
        if dataset:
            self.scan_times = dataset.scan_times
            self.lstarts = dataset.lstarts
            self.lends = dataset.lends
            self.av_bcs = dataset.av_bcs
            self.beam_offs = dataset.beam_offs
            self.beam_offs2 = dataset.beam_offs2
            self.T_vals = dataset.T_vals
            return
        lstarts, lends, T_vals, beam_offs, av_bcs, beam_offs2, beam_offs3 \
                = [], [], [], [], [], [], []
        if self.beamline == 'Polaris':
            fname_pre = 'POL'
            lflocation = r'\\isis\inst$\ndxpolaris\Instrument\data'
        elif self.beamline == 'Gem':
            fname_pre = 'GEM'
            lflocation = r'\\isis\inst$\ndxgem\Instrument\data'
        else:
            fname_pre = ''
            print('Recognised beamlines are "Polaris" or "Gem"')
        if self.filepath:
            fnames = os.listdir(self.filepath)
            log_files = [fname for fname in fnames if '.log' in fname]
            if len(log_files) > 0:
                if log_files[0][3] == 'A':
                    fname_pre += 'ARIS00' #hack for new Polaris files
                    print 'Modified Polaris log filenames'
        else:
            log_files = []
        log_fnames = [self.filepath + fname_pre + str(n) + '.log' for n in
                      self.expt_nums]
        ebo_t = None #this is for working out extra beam offs
        for i1, lf in enumerate(log_fnames):
            if re.split(r'\\|/', lf)[-1] not in log_files:
                print "%s doesn't exist. Try looking in %s for files." \
                        % (lf, lflocation)
                return
            log_data = pd.read_csv(lf, header=None, delim_whitespace=True,
                                   names=['Time', 'String', 'Value'])
            lstarts.append(np.datetime64(log_data.iloc[0, 0]))
            lends.append(np.datetime64(log_data.iloc[-1, 0]))
            if Tstring:
                if type(Tstring) != type(''):
                    if self.beamline == 'Polaris' and max(self.expt_nums)\
                       > 100000:
                        Tstring = 'Temp_2'
                    else:
                        Tstring = 'temp2'
                T_vals.append(log_data.iloc[:, 2].values\
                              [np.where(log_data.iloc[:, 1].values\
                                        == Tstring)])
            if self.beamline == 'Polaris' and max(self.expt_nums) > 100000:
                bcs = np.where(log_data.iloc[:, 1].values == \
                               'TS1_beam_current')
            else:
                bcs = np.where(log_data.iloc[:, 1].values == 'TS1')
            beam_currents = log_data.iloc[:, 2].values[bcs]
            beam_real = np.where(beam_currents > -1)
            beam_currents = beam_currents[beam_real]
            bc_times = log_data.iloc[:, 0].values[bcs][beam_real]
            start_marker = 0 #determines whether beam offs at start of log
                             #file or end (ones in the middle are ignored)
            av_bc = np.mean(np.array([float(bc) for bc in beam_currents]))
            av_bcs.append(av_bc)
            if av_bc < self.beam_min:
                beam_offs.append(self.expt_nums[i1])
                if (lends[i1] - lstarts[i1]) / np.timedelta64(1, 's') >\
                   beam_off_time:
                    beam_offs2.append((i1 + 1, lends[i1]))
            #now work out if a long time between lstart and previous lend
            if i1:
                if (lstarts[i1] - lends[i1 - 1]) / np.timedelta64(1, 's') >\
                   beam_off_time:
                    beam_offs2.append((i1, lends[i1 - 1]))
                    beam_offs2.append((i1, lstarts[i1] - \
                                      np.timedelta64(beam_off_time, 's')))
            #the below is code for partial beam_on log files
            #for i2, bc in enumerate(beam_currents):
            #    if float(bc) < self.beam_min:
            #        if i2 == 0:
            #            start_marker = 1
            #        if ebo_t:
            #            continue
            #        else:
            #            ebo_t = bc_times[i2]
            #            if i2 != 0: #i.e. beam offs won't be inserted here
            #                start_marker = 0
            #    else:
            #        if ebo_t:
            #            time_diff = (np.datetime64(bc_times[i2]) - \
            #                    np.datetime64(ebo_t)) / \
            #                    np.timedelta64(1, 's')
            #            if start_marker and time_diff > beam_off_time:
            #                beam_offs3.append((i1, [np.datetime64(ebo_t), 
            #                                np.datetime64(bc_times[i2])]))
            #            ebo_t = None
            #        else:
            #            continue
        T_vals = np.array([np.mean([float(val) for val in run]) for run in
                           T_vals])
        print '%d runs have some beam off (less than %.1f uA)' % \
                (len(beam_offs), self.beam_min)
        print 'Start time = %s' % str(lstarts[0])
        print 'End time = %s' % str(lends[-1])
        scan_times = [ls - lstarts[0] for ls in lstarts]
        for i, bo in enumerate(beam_offs2):
            scan_times.insert(i + bo[0], bo[1] - lstarts[0])
        scan_times = np.array([st / np.timedelta64(1, 's') for st in
                               scan_times]) / 3600
        self.scan_times = scan_times
        self.lstarts = lstarts
        self.lends = lends
        self.av_bcs = av_bcs
        self.beam_offs = beam_offs
        self.beam_offs2 = beam_offs2
        if Tstring:
            self.T_vals = T_vals
        return

    def get_run_numbers(self):
        """Return array of run numbers (accounting for beam offs)"""
        run_nums = self.expt_nums[:]
        for i, bo in enumerate(self.beam_offs2):
            if i + bo[0] >= len(run_nums):
                run_nums.append(run_nums[-1] + 0.5)
            else:
                run_nums.insert(i + bo[0],  run_nums[i + bo[0]] - 0.5)
        return np.array(run_nums)

    def get_expt_fnames_all(self): 
        if self.tof:
            suffix = '_b' + str(self.bank_num) + '_TOF.dat'
        else:
            suffix = '_b' + str(self.bank_num) + '_D.dat'
        if self.beamline == 'Polaris':
            if max(self.expt_nums) < 100000:
                return [self.filepath + 'pol' + str(n) + suffix for n in 
                        self.expt_nums]
            else:
                if self.tof:
                    suffix = '-b_' + str(self.bank_num) + '-TOF.dat'
                else:
                    suffix = '-b_' + str(self.bank_num) + '-d.dat'
                return [self.filepath + 'POL' + str(n) + suffix for n in 
                        self.expt_nums]
        elif self.beamline == 'Gem':
            return [self.filepath + 'GEM' + str(n) + suffix for n in 
                    self.expt_nums]
        else:
            return

    def get_expt_fnames(self, print_missing=True):
        result = []
        missing = []
        fnames_all = self.get_expt_fnames_all()
        file_list = [self.filepath + fl for fl in os.listdir(self.filepath)]
        for i, f in enumerate(fnames_all):
            if f in file_list:
                result.append(f)
            else:
                result.append('')
                missing.append(self.expt_nums[i])
                if print_missing:
                    print 'File %d is missing' % self.expt_nums[i]
                    print f
        return result

    def get_data(self, print_missing=True, dataset=None):
        """Get diffraction data

        Args:
            print_missing (bool): whether to print the missing filenames
            dataset: if specified then borrows data from other dataset
        """
        if dataset:
            self.data = dataset.data
            return
        data = []
        first_missing = False
        expt_fnames = self.get_expt_fnames(print_missing=print_missing)
        bo_indices = [self.expt_nums.index(bo) for bo in self.beam_offs]
        expt_fnames = ['' if i in bo_indices else fname for i, fname in
                       enumerate(expt_fnames)]
        for i, bo in enumerate(self.beam_offs2):
            expt_fnames.insert(i + bo[0],  '')
        marker = 0
        bo_indices2 = [bo[0] + i for i, bo in 
                       enumerate(self.beam_offs2)]
        i1 = -1 #this refers to index of beam_offs not beam_offs2
        for i, f in enumerate(expt_fnames):
            i1 += 1
            if f:
                marker = i
                data.append(pd.read_csv(f, header=None, 
                                        delim_whitespace=True,
                                        names=['x', 'y', 'e']))
            else:
                if i1 not in bo_indices and i not in bo_indices2:
                    self.beam_offs.append(self.expt_nums[i1])
                    self.beam_offs.sort()
                    print "File %s missing, %d added to beam offs" \
                            % (f, self.expt_nums[i1])
                if len(data):
                    data.append(pd.DataFrame({'x' : data[0]['x'].values, 
                                              'y' : \
                                              np.zeros(data[0].shape[0]),
                                              'e' : \
                                              np.zeros(data[0].shape[0])}))
                else:
                    first_missing=True
            if i in bo_indices2:
                i1 -= 1
        if first_missing == True and len(data) > 0:
            data.insert(0, 
                        pd.DataFrame({'x' : data[marker - 1]['x'].values, 
                                      'y' : \
                                      np.zeros(data[marker - 1].shape[0]),
                                      'e' : \
                                    np.zeros(data[marker - 1].shape[0])}))
        if len(data) == 0:
            print "No datasets found"
        self.data = data
        return

    def get_igan_data(self, igan_number, filepath_igan=None):
        """Return IGAn data for given sample number

        Args:
            igan_number: if not an integer then must be another dataset
            with igan_data already associated with it.
            filepath_igan: used to specify igan filepath if different to
            normal filepath.
        """
        if type(igan_number) != int:
            self.igan_data = igan_number.igan_data
            return
        if len(self.lstarts) == 0:
            print 'You need to run Dataset.get_scan_times() before \
                    Dataset.get_igan_data()'
            return
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 
                  'Sep', 'Oct', 'Nov', 'Dec']
        igan_number = ''.join(['0'] * (4 - len(str(igan_number)))) + \
                str(igan_number)
        if not filepath_igan:
            filepath_igan = self.filepath
        igan_fpath = filepath_igan + 'Sample_' + igan_number + '/'
        igan_dtimes = []
        igan_types = []
        igan_datasets = []
        with open(igan_fpath + 'Sample Log.txt', 'r') as f:
            for l in f:
                lsplit = l.split()
                if 'begins' in lsplit:
                    m = str(months.index(lsplit[3][:3]) + 1)
                    d = lsplit[2]
                    if len(m) == 1:
                        m = '0' + m
                    if len(d) == 1:
                        d = '0' + d
                    igan_dtimes.append('T'.join(['-'.join([lsplit[4], m, 
                                                           lsplit[2]]), 
                                                 lsplit[0]]))
                    igan_types.append(lsplit[7:9])
        igan_times =  [(np.datetime64(idt) - self.lstarts[0]) / 
                       (np.timedelta64(1, 's') * 3600) for idt in 
                       igan_dtimes]
        for i, igan_run in enumerate(igan_types):
            it_fname = igan_fpath + igan_run[0] + '/' + igan_run[0] + \
                    igan_run[1] + '/' + 'Data.txt'
            try:
                with open(it_fname, 'r') as f:
                    for i1, l in enumerate(f):
                        if l.lstrip().rstrip() == '':
                            it_header = i1
                            break
            except IOError:
                print("%s %s is lacking Data.txt therefore ignored"\
                      % (igan_run[0], igan_run[1]))
                continue
            igan_dataset = pd.read_csv(it_fname, header=it_header, 
                                       delim_whitespace=True, 
                                       usecols=[0, 1, 2, 3]).values[:-1, :]
            igan_dataset[:, 0] = igan_dataset[:, 0].astype(np.float) / 60 +\
                    igan_times[i]
            igan_datasets.append(igan_dataset)
        igan_data = np.row_stack(igan_datasets)
        self.igan_data = igan_data.astype('float64')
        return

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
            i0, i1 = [np.searchsorted(tth[:, 0], tthval) for tthval in
                      tth_range]
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
            filepath (str): file path for destination files (defaults to 
            current directory).
            pre_fname (str): precursor string for file names
            post_fname (str): post string for file names (defaults to 
            '.dat')
            sep (str): separator string for columns (defaults to 
            '\t'---tab)
            file_nums (list): list of numbers, defaults to index numbers 
            of Dataset.
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
                t = np.array(range(len(self.data)))
            ti = np.abs(t - tval).argmin()
            self.data[ti].to_csv(fnames[0], index=False, header=False, 
                                 sep=sep)
            return
        for i in range(len(self.data)):
            self.data[i].to_csv(fnames[i], index=False, header=False, 
                                sep=sep)
        return        
    
    def sum_dsets(self, sum_num, file_range=None, t=None, T=None):
        """Return mean summed datasets for sum_num interval
        Args:
            sum_num (int): number of scans to average over
            file_range: file range (list of two inclusive file numbers) 
            within which to sum data.
            t (arr): time array
            T (arr): temperature array
        Returns:
            result: Dataset of summed/averaged datasets with propogated
            errors (result.scan_times will include averaged scan times).
            t_result: average time array if t != None.
            T_result: average temperature array if T != None
        """
        if file_range:
            indices = [self.expt_nums.index(fn) for fn in file_range]
            bo_indices = np.array([b[0] for b in self.beam_offs2])
            idxs = [i + np.searchsorted(bo_indices, i) for i in indices]
        else:
            indices = [0, len(self.expt_nums) - 1]
            idxs = [0, len(self.data) - 1]
        result = []
        for i, dset in enumerate(self.data[idxs[0]:idxs[1] + 1]):
            if i % sum_num == 0:
                if i:
                    if i == len(self.data[idxs[0]:idxs[1] + 1]) - 1:
                        new_y = np.column_stack((new_y, dset['y'].values))
                        new_e = np.column_stack((new_e, 
                                                 dset['e'].values**2))
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
        if t is not None:
            t = t[idxs[0]:idxs[1] + 1]
            t_result = self._sum_mean(t, idxs, sum_num)
        if T is not None:
            T = T[idxs[0]:idxs[1] + 1]
            T_result = self._sum_mean(T, idxs, sum_num)
        res = Dataset(self.filepath, self.expt_nums[indices[0]], 
                      self.expt_nums[indices[1]], self.beamline, 
                      self.beam_min, self.bank_num, self.tof)
        res.data = result
        if type(self.scan_times) == type(np.array([])) and\
           len(self.scan_times):
            new_st = self._sum_mean(self.scan_times[idxs[0]:idxs[1] + 1],
                                    idxs, sum_num)
            res.scan_times = np.array(new_st)
        if type(self.T_vals) == type(np.array([])) and len(self.T_vals):
            new_T = self._sum_mean(self.T_vals[idxs[0]:idxs[1] + 1], idxs,
                                   sum_num)
            res.T_vals = np.array(new_T)
        if T is None and t is None:
            return res
        elif T is not None and t is None:
            return res, T_result
        elif t is not None and T is None:
            return res, t_result
        else:
            return res, t_result, T_result

    def plot(self, tval, t=None, xlabel=u'd / \u00C5', 
             ylabel='Intensity / Counts', figsize=(10, 7), x_range=None, 
             y_range=None, linecolour=None, labels=None, legend=True,
             legend_loc=0, xclip=True, normalize=False, 
             waterfall_offset_x=0, waterfall_offset_y=0):
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
        Returns:
            fig: figure instance
            ax: axes instance
        """
        if t is None:
            t = np.array(range(len(self.data)))
        if type(tval) == int or type(tval) == float:
            tval = [tval]
        if type(linecolour) == str:
            linecolour = [linecolour]
        if type(labels) == type(None):
            labels = [str(tv) for tv in tval]
        tis = [np.abs(t - tv).argmin() for tv in tval]
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
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
            if type(linecolour) == type(None):
                ax.plot(data_x, data_y, label=labels[i])
            else:
                ax.plot(data_x, data_y, color=linecolour[i], 
                        label=labels[i])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if legend:
            ax.legend(loc=legend_loc)
        if type(x_range) != type(None):
            ax.set_xlim(x_range[0], x_range[1])
        if type(y_range) != type(None):
            ax.set_ylim(y_range[0], y_range[1])
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
            t = np.array(range(len(self.data)))
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
                           labels=None, legend=True, legend_loc=0):
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
        Returns:
            fig: figure instance
            ax: axes instance
        """
        if t is None:
            t = np.array(range(len(self.data)))
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
            m, c = np.polyfit(np.concatenate((data_x[:norm_pts], 
                                              data_x[-norm_pts:])),
                              np.concatenate((data_y[:norm_pts], 
                                              data_y[-norm_pts:])), 1)
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

    def plotQ(self, tval, t=None, xlabel=u'Q / \u00C5$^{-1}$', 
              ylabel='Intensity / Counts', figsize=(10, 7), x_range=None, 
              y_range=None, linecolour='g'):
        """Return a 2D plot of the diffraction data
        
        Args:
            tval: which time/run number to plot
            t: defaults to range(len(data))
            xlabel: label for x-axis
            ylabel: label for y-axis
            figsize: size of figure (inches by inches)
            x_range (list): x range
            y_range (list): y range
            linecolour (str): colour of plotted line

        Returns:
            fig: figure instance
            ax: axes instance
        """
        if type(t) == type(None):
            t = np.array(range(len(self.data)))
        ti = np.abs(t - tval).argmin()
        data_x = 2 * np.pi / self.data[ti]['x'].values
        data_y = self.data[ti]['y'].values
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        ax.plot(data_x, data_y, color=linecolour)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if type(x_range) != type(None):
            ax.set_xlim(x_range[0], x_range[1])
        if type(y_range) != type(None):
            ax.set_ylim(y_range[0], y_range[1])
        fig.tight_layout()       
        return fig, ax
        
    def contour_plot(self, t=None, xlabel='Run number', 
                     ylabel=u'd / \u00C5', zlabel='Intensity / Counts', 
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
            cbar: colourbar instance
        """
        #26/01/16 rewrite
        #data_y, data_z = self.data_xy() #data_y is 2theta, data_z is intensity
        if type(t) == type(None):
            t = self.scan_times
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
        plt.tight_layout()
        return fig, ax, cbar
        
    def contour_temp(self, T, t=None, xlabel='Run number', 
                     ylabel=u'd / \u00C5', ylabel2=u'Temperature / \u00B0C',
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
            zscale: 'log' for log scaling and 'sqrt' for square root
            log_zlabel (str): Title of colourbar when zscale='log'
            sqrt_zlabel (str): Title of colourbar when zscale='sqrt'

        Returns:
            fig: figure instance
            ax1: temperature line plot axes
            ax2: contour plot axes
            cbar: colourbar instance
        """
        data_y, data_z = self.data_xy() #data_y is 2theta, data_z is I(2th)
        if type(t) == type(None):
            t = np.meshgrid(np.arange(data_y.shape[1]), np.arange(data_y.shape[0]))[0]
        elif t.ndim == 1:
            t = np.meshgrid(t, np.arange(data_y.shape[0]))[0]
        if x_range:
            i0, i1 = [np.abs(t[0, :] - val).argmin() for val in x_range]
            t = t[:, i0:i1 + 1]
            data_y = data_y[:, i0:i1 + 1]
            data_z = data_z[:, i0:i1 + 1]
        if y_range:
            i0, i1 = [np.abs(data_y[:, 0] - val).argmin() for val in y_range]
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
        ax1.plot(t[0, :], T, color=Tcolour)
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
        return fig, ax1, ax2, cbar
    
    def contour_igan(self, xlabel='Time / h', ylabel=u'd / \u00C5', 
                     ylabel2=u'Temperature / \u00B0C', ylabel3='Mass / mg',
                     ylabel4='Pressure / mbar', zlabel='Intensity / Counts',
                     colour_num=20, figsize=(10, 10), x_range=None, 
                     y_range=None, z_range=None, xyflip=False, Tcolour='g', 
                     masscolour='r', pressurecolour='b', 
                     height_ratios=[1, 1, 1, 2], zscale=None, 
                     log_zlabel='log(Intensity / Counts)',
                     sqrt_zlabel = '$\sqrt{Intensity / Counts}$', 
                     T_range=None, m_range=None, p_range=None,
                     plot_run_nums=False, run_num_ticks=5):
        """Return a contour plot of the data
        
        Args:
            xlabel: label for x axis
            ylabel: label for y axis
            ylabel: label for temperature plot
            zlabel: label for colour bar
            colour_num: number of colours in colour bar
            figsize: size of the figure (inches by inches)
            x_range (list): x range
            y_range (list): y range
            z_range (list): z range
            T_range (list): Temperature range
            m_range (list): mass range
            p_range (list): pressure range
            height_ratios: ratios of heights of subplots
            zscale: can be 'log' or 'sqrt'
            plot_run_nums (bool): whether to plot run numbers as second
            x axis or not.
            run_num_ticks: number of ticks if plot_run_nums is True

        Returns:
            fig: figure instance
            ax_cont: contour plot axes
            ax_T: Temperature axes
            ax_m: mass axes
            ax_p: pressure axes
            cbar: colourbar instance
        """
        data_y, data_z = self.data_xy() #data_y is 2theta, data_z is I(2th)
        igan = self.igan_data
        t = self.scan_times
        igan_t, igan_T, igan_m, igan_p = [igan[:, 0], igan[:, 3], 
                                          igan[:, 1], igan[:, 2]]
        if t.ndim == 1:
            t = np.meshgrid(t, np.arange(data_y.shape[0]))[0]
        if plot_run_nums:
            t2 = self.get_run_numbers()
        if x_range:
            i0, i1 = [np.abs(t[0, :] - val).argmin() for val in x_range]
            t = t[:, i0:i1 + 1]
            data_y = data_y[:, i0:i1 + 1]
            data_z = data_z[:, i0:i1 + 1]
            if plot_run_nums:
                t2 = t2[i0:i1]
        if y_range:
            i0, i1 = [np.abs(data_y[:, 0] - val).argmin() for val in y_range]
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
        gs = mpl.gridspec.GridSpec(4, 1, height_ratios=height_ratios)
        ax_cont = fig.add_subplot(gs[3, 0])
        ax_T = fig.add_subplot(gs[1, 0], sharex=ax_cont)
        ax_m = fig.add_subplot(gs[2, 0], sharex=ax_cont)
        ax_p = fig.add_subplot(gs[0, 0], sharex=ax_cont)
        cont = ax_cont.contourf(t, data_y, data_z, colour_num)
        ax_cont.set_xlabel(xlabel)
        ax_cont.set_ylabel(ylabel)
        ax_cont.tick_params(which='both', top=False, right=False, 
                            direction='out')
        ax_cont.set_xlim(t[0, 0], t[0, -1])
        axins = inset_axes(ax_cont, width='5%', height='100%', loc=6,
                           bbox_to_anchor=(1.05, 0., 1, 1), borderpad=0,
                           bbox_transform=ax_cont.transAxes)
        cbar = plt.colorbar(cont, cax=axins)
        if zscale == 'log':
            cbar.set_label(log_zlabel, rotation=270, labelpad=20)
        elif zscale == 'sqrt':
            cbar.set_label(sqrt_zlabel, rotation=270, labelpad=30)
        else:
            cbar.set_label(zlabel, rotation=270, labelpad=20)
        ax_T.plot(igan_t, igan_T, color=Tcolour)
        ax_T.set_ylabel(ylabel2)
        ax_T.tick_params(which='both', top=False, right=False, 
                         direction='out')
        if T_range:
            ax_T.set_ylim(T_range)
        plt.setp(ax_T.get_xticklabels(), visible=False)
        ax_m.plot(igan_t, igan_m, color=masscolour)
        ax_m.set_ylabel(ylabel3)
        ax_m.tick_params(which='both', top=False, right=False, 
                         direction='out')
        if m_range:
            ax_m.set_ylim(m_range)
        plt.setp(ax_m.get_xticklabels(), visible=False)
        ax_p.plot(igan_t, igan_p, color=pressurecolour)
        ax_p.set_ylabel(ylabel4)
        ax_p.tick_params(which='both', top=False, right=False, 
                         direction='out')
        if p_range:
            ax_p.set_ylim(p_range)
        plt.setp(ax_p.get_xticklabels(), visible=False)
        if plot_run_nums:
            ax_rn = ax_cont.twiny()
            ax_rn.set_xlim(ax_cont.get_xlim())
            ax_rn.set_xlabel('Run numbers')
            tick_is = range(0, len(t2), len(t2) / run_num_ticks)
            rn_labels = [str(t2[i]) for i in tick_is]
            rn_tick_pos = [t[0][i] for i in tick_is]
            ax_rn.set_xticks(rn_tick_pos)
            ax_rn.set_xticklabels(rn_labels)
        fig.tight_layout(rect=(0, 0, 0.85, 1))
        if plot_run_nums:
            return fig, ax_cont, ax_m, ax_T, ax_p, ax_rn, cbar
        return fig, ax_cont, ax_m, ax_T, ax_p, cbar
        
    def contour_mult(self, T=None, t=None, xlabel='Run number', 
                     ylabel=u'd / \u00C5', 
                     ylabel2=u'Temperature / \u00B0C',
                     zlabel='Intensity / Counts', colour_num=20, 
                     figsize=(10, 7), x_range=None, y_range=None, 
                     z_range=None, xyflip=False, Tcolour='g', 
                     height_ratios=None, width_ratios=None, zscale=None, 
                     log_zlabel='log(Intensity / Counts)',
                     sqrt_zlabel = '$\sqrt{Intensity / Counts}$', 
                     grid=None, sharey=True, shareT=True, colourbar=False):
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
            axes: list of all axes instances (includes inserted axis
            and colourbar instance per contour plot if colourbar is True).
             
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
            t = np.meshgrid(np.arange(data_y.shape[1]), np.arange(data_y.shape[0]))[0]
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
            ax.tick_params(which='both', top=False, right=False, 
                           direction='out')
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
            axes.append(ax)
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
                axes.append(axins)
                axes.append(cbar)
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
            t = np.array(range(len(self.data)))
        ti_start, ti_end = [np.abs(t - tval).argmin() for tval in t_range]
        t_indices = range(ti_start, ti_end + 1, 1)
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
        return
    
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
            t = np.meshgrid(np.arange(data_y.shape[1]), np.arange(data_y.shape[0]))[0]
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
        return
     
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
            print dset.shape
    def _get_shapes(self):
        """Used for debugging purposes. Returns shapes of all datasets"""
        return [dset.shape for dset in self.data]

def get_expt_numbers(first_file_number, last_file_number):
    """Return list of experiment numbers"""
    return range(first_file_number, last_file_number + 1, 1)

def get_file_list(filepath):
    """Return list of files at filepath"""
    result = os.listdir(filepath)
    return [filepath + fl for fl in result]

def get_expt_fnames_all(filepath, expt_numbers, fname_pre='pol', 
                        file_extension=None, bank_num=5, tof=True):
    """Return all filenames assuming no beam offs
    
    Args:
        filepath (str): file path
        expt_numbers (list): list of experiment numbers
        fname_pre (str): defaults to 'pol' for Polaris reduced data
        file_extension: if set to string, then overrides default bank_num and
        tof args.
        bank_num (int or str): which bank number's data to load
        tof (bool): whether to load the data in TOF or d.
    """
    if not file_extension:
        if tof:
            file_extension = '_b' + str(bank_num) + '_TOF.dat'
        else:
            file_extension = '_b' + str(bank_num) + '_D.dat'
    return [filepath + fname_pre + str(n) + file_extension for n in expt_numbers]

def get_expt_fnames(filepath, expt_numbers, fname_pre='pol', file_extension=None,
                    bank_num=5, tof=True, missing_nums=False, print_missing=True):
    """Return all expt file names which exist in directory
    
    Args:
        filepath (str): file path
        expt_numbers (list): list of experiment numbers
        fname_pre (str): defaults to 'pol' for Polaris reduced data
        file_extension: if set to string, then overrides default bank_num and
        tof args.
        bank_num (int or str): which bank number's data to load
        tof (bool): whether to load the data in TOF or d.
        missing_nums (bool): whether to return list of missing numbers
        print_missing (bool): whether to print missing files
    """
    result = []
    missing = []
    fnames_all = get_expt_fnames_all(filepath, expt_numbers, fname_pre,
                                     file_extension, bank_num, tof)
    file_list = get_file_list(filepath)
    for i, f in enumerate(fnames_all):
        if f in file_list:
            result.append(f)
        else:
            result.append('')
            missing.append(expt_numbers[i])
            if print_missing:
                print 'File %d is missing' % expt_numbers[i]
                print f
    if missing_nums:
        return result, missing
    else:
        return result

def plotQ(tval, datasets=None, t=None, first_file_number=None, Q=True,
           last_file_number=None, filepath=None, fname_pre='pol', file_extensions=None,
           bank_nums=5, xlabel=u'Q / \u00C5$^{-1}$', 
           ylabel='Intensity / Counts', figsize=(10, 7), x_range=None, y_range=None,
           linecolours=['g', 'r', 'b', 'm', 'c'], legend=True, legend_loc=0,
           stack_shifts=None, normalize=False):
    """Return a 2D plot of the diffraction data in Q
        
    Args:
        tval: which time/run number to plot
        datasets: list of datasets from which to plot
        t: defaults to range(len(data))
        first_file_number (int): first file number
        Q (bool): if True (default) then plot in Q, else in d.
        last_file_number (int): last file number
        filepath (str): file path
        expt_numbers (list): list of experiment numbers
        fname_pre (str): defaults to 'pol' for Polaris reduced data
        file_extensions (list): if set then overrides default bank_num args.
        bank_nums (int or list): which bank number's data to load - also acts as
        labels for graph legend
        xlabel (str): label for x-axis
        ylabel (str): label for y-axis
        figsize (list): size of figure (inches by inches)
        x_range (list): x range
        y_range (list): y range
        linecolours (list): colours of plotted line as strings
        legend (bool): determines presence of legend or otherwise
        legend_loc: determines position of legend
        stack_shifts (list): list of y fractions to shift successive datasets in y
        normalize (bool): normalizes in y if True (False by default)
        """
    if type(t) == type(None):
        ti = tval
    else:
        ti = np.abs(t - tval).argmin()
    if type(bank_nums) not in [tuple, list]:
        bank_nums = [bank_nums]
    if not datasets:
        if type(first_file_number) == type(None) or type(last_file_number) == type(None):
            raise TypeError('first_file_number/last_file_number or datasets not specified')
        if type(filepath) == type(None):
            raise TypeError('filepath has not been specified')
        datasets = []
        expt_number = range(first_file_number, last_file_number + 1)[ti]
        expt_fnames = []
        if type(file_extensions) == type(None):
            file_extensions = len(bank_nums) * [None]
        elif type(file_extensions) == str:
            file_extensions = len(bank_nums) * [file_extensions]
        for i, bn in enumerate(bank_nums):
            ef = get_expt_fnames(filepath, [expt_number], 
                                 file_extension=file_extensions[i], 
                                 bank_num=bank_nums[i], tof=False, 
                                 print_missing=True)
            expt_fnames += ef
        for i, f in enumerate(expt_fnames):
            datasets.append(pd.read_csv(f, header=None, delim_whitespace=True,
                            names=['x', 'y', 'e']))
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    if stack_shifts:
        ss_factors = []
    for i, d in enumerate(datasets):
        if Q:
            data_x = 2 * np.pi / d['x'].values
        else:
            data_x = d['x'].values
        data_y = d['y'].values
        if normalize:
            data_y = (data_y - data_y.min()) / (data_y.max() - data_y.min())
        if stack_shifts:
            if i < len(datasets) - 1:
                ss_factors.append(stack_shifts[i] * (data_y.max() - data_y.min()))
            if i:
                data_y += np.sum(np.array(ss_factors[:i]))
        ax.plot(data_x, data_y, color=linecolours[i], label=bank_nums[i])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if legend:
        ax.legend(loc=legend_loc)
    if type(x_range) != type(None):
        ax.set_xlim(x_range[0], x_range[1])
    if type(y_range) != type(None):
        ax.set_ylim(y_range[0], y_range[1])
    plt.tight_layout()  

