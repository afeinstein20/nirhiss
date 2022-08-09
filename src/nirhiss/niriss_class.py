import os
import numpy as np
from astropy.io import fits
import time as time_pkg

from .tracing import mask_method_edges, mask_method_ears, ref_file
from .masking import data_quality_mask, interpolating_col
from .background_utils import *
from .extraction import (dirty_mask, box_extract, optimal_extraction_routine)


__all__ = ['ReduceNirHiss']


class ReduceNirHiss(object):

    def __init__(self, files, f277_filename,
                 crds_cache=None, data_dir=None, output_dir=None):
        """
        Initializes the NIRISS S3 data reduction class.

        Parameters
        ----------
        filename : str
           The name of the FITS file with NIRISS observations.
        data_path : str, optional
           The path to where the input FITS files are stored. Default
           is None. If None, will search the current working directory
           for the files.
        output_dir : str, optional
           The path where output files will be saved. Default is None.
           if None, will save all files to the current working directory.

        Attributes
        ----------
        filename : str
           The science file name.
        data_path : str
           The path to where the science file is stored.
        output_dir : str
           The path to where output files will be stored.
        bkg : np.ndarray
           Array of background values.
        bkg_var : np.ndarray
           Array of variance estimations on the background.
        cr_mask : np.ndarray
           Array of masks for bad pixels.
        bkg_removed : np.ndarray
           Data - background array.
        trace_table : astropy.table.Table
           Astropy table with x,y coordinates for each order.
        trace_type : str
           Reference to which method of trace identification was used.
        box_spectra_order1 : np.ndarray
           Box extracted spectra for the first order.
        box_spectra_order2 : np.ndarray
           Box extracted spectra for the second order.
        box_mask_separate : np.ndarray
           Attribute for separate box masks per each order. Created
           when `dirty_mask(return_together == False)`.
        """
        self.files = files
        self.datetime = time_pkg.strftime('%Y-%m-%d')

        if data_dir is not None:
            self.data_dir = data_dir
        else:
            self.data_dir = os.getcwd()

        if output_dir is not None:
            self.output_dir = output_dir
        else:
            self.output_dir = os.getcwd()

        # Opens the science FITS file and sets up all proper
        #   data attributes for the reduction steps.
        self.setup()
        self.setup_f277(f277_filename)

        self.bkg = None
        self.box_mask = None
        self.box_var1 = None
        self.box_var2 = None
        self.bq_masked = False
        self.trace_type = None
        self.trace_table = None
        self.bkg_removed = None
        self.box_mask_separate = None
        self.box_spectra_order1 = None
        self.box_spectra_order2 = None

        return

    def setup(self):
        """Sets up all proper attributes from the FITS file.

        Attributes
        ----------
        times : np.ndarray
           Array of time values based on the exposure start
           and stop times indicated in the FITS header.
        time_unit : str
           What units the time array is in.
        data : np.ndarray
           3D array of science group images.
        errors : np.ndarray
           3D array of errors associated with the science
           frames.
        dq : np.ndarray
           3D array of data quality masks, indicating the
           location of bad pixels.
        variance : np.ndarray
           3D array of poisson estimated variances.
        """
        for fn in self.files:
            hdus = []
            counter = []
            for i in range(len(files)):
                hdus.append(fits.open(os.path.join(self.data_dir,
                                                   self.files[i])))
                counter.append(len(hdus[i][1].data))

            counts = np.nansum(counter)

        self.data = np.zeros((counts, 256, 2048))
        self.variance = np.zeros((counts, 256, 2048))
        self.errors = np.zeros((counts, 256, 2048))
        self.times = np.zeros(counts)
        self.dq = np.zeros((counts, 256, 2048))

        for i in range(len(counter)):
            inttime = hdus[i][0].header['EFFINTTM']

            if i == 0:
                start, end = 0, counter[0]
            else:
                start = np.nansum(counter[:i])
                end = np.nansum(counter[:i+1])

            self.data[start:end] = np.copy(hdus[i][1].data)
            self.errors[start:end] = np.copy(hdus[i][2].data)
            self.variance[start:end] = hdus[i][5].data * inttime**2.0

            # makes DQ mask from DQ fits extension
            self.dq[start:end] = masking.data_quality_mask(hdus[i][3].data)

            self.times[start:end] = np.copy(hdus[i][4].data['int_mid_BJD_TDB'])
            hdus[i].close()

        self.times += 0.5
        self.time_unit = 'BJD'
        return

    def setup_f277(self, filename):
        """Opens and assigns proper attributes for the F277W filter
        observations.

        Parameters
        ----------
        filename : str
           The name of the F277W FITS file.

        Attributes
        ----------
        f277 : np.ndarray
           Science images from this filter.
        """
        with fits.open(os.path.join(self.data_dir, filename)) as hdu:
            self.f277 = hdu[1].data
            self.f277_dq = data_quality_mask(hdu[3].data)
        return

    def clean_up(self):
        """
        The `clean_up` routine interpolated over bad pixel values,
        which are marked in the data quality images (`self.dq`).
        This routine removes bad quality pixels from the following
        images:
           - `self.data`
           - `self.errors`
           - `self.variance`
        """
        print('Cleaning data from bad DQ pixels . . .')
        for i in tqdm(range(len(self.data))):
            self.data[i] = interpolating_image(self.data[i], self.dq[i])
            self.errors[i] = interpolating_image(self.errors[i], self.dq[i])
            self.variance[i] = interpolating_image(self.variance[i], self.dq[i])

        self.dq_masked = True

        self.median = np.nanmedian(self.data, axis=0)

    def model_bkg_removal(self, filename, data_dir=None):
        """
        Scales and removes the pre-determined background model from the STScI
        crew. The models can be downloaded here:

        Parameters
        ----------
        filename : str
           Name of the background model file. Should be a `.npy` file.
        data_dir : str, optional
           The path to where the background model is stored. Default is `None`.
           If `None` will search the directory where the FITS files are stored
           (`self.data_dir`).

        Attributes
        ----------
        scaled_model_bkg : np.ndarray
        """

    def map_trace(self, method='profile', ref_filename=None, isplots=0):
        """
        Calculates the trace of Orders 1, 2, and 3 of NIRISS.

        Parameters
        ----------
        method : str, optional
           Decision on which trace extraction routine to run.
           Options are: `edges` (uses a canny-edge detection
           routine), `centers` (uses the spatial profile), and
           `ref` (uses the STScI JWST reference frame).
        ref_filename : str, optional
           The name of the reference frame containing the order
           trace x,y position values. Default is None. This is
           a required parameter if you are running `method=='ref'`.

        Attributes
        ----------
        trace_ear : astropy.table.Table
           Astropy table with x,y coordinates for each order.
           `trace_ear` is initialized when using the method `edges`.
        trace_edge : astropy.table.Table
           Astropy table with x,y coordinates for each order.
           `trace_edge` is initialized when using the method `centers`.
        """
        if method.lower() == 'edges':
            if self.f277 is not None:
                self.trace_table = mask_method_edges(self, isplots=isplots)
                self.trace_type = 'ear'
            else:
                return('Need F277W filter to run this trace finding method.')

        elif method.lower() == 'profile':
            self.trace_table = mask_method_ears(self, isplots=isplots)
            self.trace_type = 'edge'

        elif method.lower() == 'ref':
            self.trace_table = ref_file(ref_filename)
            self.trace_type = 'reference'

        else:
            return('Trace method not implemented. Options are `edges` and '
                   '`centers`.')

    def create_box_mask(self, boxsize1=60, boxsize2=50, boxsize3=40,
                        booltype=True, return_together=True):
        """Creates a box mask to extract the first and second NIRISS orders.

        Can set different box sizes for each order and also return a single
        mask with both orders (`return_together==True`) or return masks for
        each order (`return_together==False`).

        Parameters
        ----------
        boxsize1 : int, optional
           Box size for the first order. Default is 60 pixels.
        boxsize2 : int, optional
           Box size for the second order. Default is 50 pixels.
        booltype : bool, optional
           Sets the dtype of the mask array. Default is True
           (returns array of boolean values).
        return_together : bool, optional
           Determines whether or not to return one combined
           box mask or masks for both orders. Default is True
           (returns 2 separate masks).

        Attributes
        ----------
        boxsize1 : int
           The box size for the first order.
        boxsize2 : int
           The box size for the second order.
        box_mask : np.ndarray
           Attribute for a combined box mask per each order. Created
           when `return_together == True`.
        box_mask_separate : np.ndarray
           Attribute for separate box masks per each order. Created
           when `return_together == False`.
        """
        if self.trace_edge is not None:
            t = self.trace_edge
        elif self.trace_ear is not None:
            t = self.trace_ear
        else:
            return('Need to run the trace identifier to create the box mask.')

        out = dirty_mask(self.median, t,
                         boxsize1=boxsize1,
                         boxsize2=boxsize2,
                         boxsize3=boxsize3,
                         booltype=booltype,
                         return_together=return_together)
        if return_together:
            self.box_mask = out
        else:
            self.box_mask_separate = np.array(out)

        self.boxsize1 = boxsize1
        self.boxsize2 = boxsize2
        self.boxsize3 = boxsize3

    def extract_box_spectrum(self):
        """
        Extracts spectra using the box mask.

        Attributes
        ----------
        box_spectra1 : np.ndarray
           Box extracted spectra for the first order.
        box_spectra2 : np.ndarray
           Box extracted spectra for the second order.
        box_var1 : np.ndarray
           Box extracted variance for the first order.
        box_var2 : np.ndarray
           Box extracted variance for the second order.
        """
        if self.box_mask_separate is None:
            self.create_box_mask(return_together=False,
                                 booltype=False)

        if self.bkg_removed is not None:
            d = np.copy(self.bkg_removed)
        else:
            d = np.copy(self.data)

        s, v = box_extract(d, self.var, self.box_mask_separate)

        self.box_var1 = np.copy(v[0])
        self.box_var2 = np.copy(v[1])
        self.box_var3 = np.copy(v[2])

        self.box_spectra1 = np.copy(s[0])
        self.box_spectra2 = np.copy(s[1])
        self.box_spectra3 = np.copy(s[2])

        return

    def fit_background(self, readnoise=11, sigclip=[4, 4, 4],
                       box=(5, 2), filter_size=(2, 2),
                       bkg_estimator=['median', ], test=True):
        """Subtracts background from non-spectral regions.

        Parameters
        ----------
        readnoise : float, optional
            An estimation of the readnoise of the detector.
            Default is 11.
        sigclip : list, array; optional
            A list or array of len(n_iiters) corresponding to the
            sigma-level which should be clipped in the cosmic
            ray removal routine. Default is [4, 4, 4].
        box : list, array; optional
            The box size along each axis. Box has two elements: (ny, nx). For
            best results, the box shape should be chosen such that the data
            are covered by an integer number of boxes in both dimensions.
            Default is (5, 2).
        filter_size : list, array; optional
            The window size of the 2D median filter to apply to the
            low-resolution background map. Filter_size has two elements:
            (ny, nx). A filter size of 1 (or (1,1)) means no filtering.
            Default is (2, 2).
        bkg_estimator : list, array; optional
            The value which to approximate the background values as. Options
            are "mean", "median", or "MMMBackground". Default is ['median', ].
        test : bool, optional
            Evaluates the background across fewer integrations to test and
            save computational time. Default is False.

        Returns
        -------
        self.bkg : np.ndarray
            The fitted background array.
        self.bkg_var : np.ndarray
            Errors on the fitted backgrouns.
        self.cr_mask : np.ndarray
            Array of masked bad pixels.
        self.bkg_removed : np.ndarray
            self.cr_mask - self.bkg after running through
            lib.masking.interpolating_image.
        """
        if self.box_mask is None:
            self.create_box_mask(return_together=True, booltype=True)

        if test is True:
            ind = 5
        else:
            ind = len(self.data)

        bkg, bkg_var, cr_mask = fitbg3(self.data[:ind], ~self.box_mask,
                                       readnoise=readnoise, sigclip=sigclip,
                                       bkg_estimator=bkg_estimator, box=box,
                                       filter_size=filter_size, inclass=True)

        self.bkg = np.copy(bkg)
        self.bkg_var = np.copy(bkg_var)
        self.bkg_removed = cr_mask - bkg

        m = np.zeros(cr_mask.shape)
        x, y, z = np.where(np.isnan(cr_mask))
        m[x, y, z] = 1
        self.cr_mask = np.copy(m)

        m = np.zeros(self.bkg_removed.shape)
        x, y, z = np.where(np.isnan(self.bkg_removed))
        m[x, y, z] = 1
        self.bkg_removed = interpolating_image(self.bkg_removed,
                                               mask=m)

    def optimal_extraction(self, proftype='median', sigma=20, Q=1.8,
                           per_quad=True, test=False, isplots=3):
        """Runs the optimal extraction routine for the NIRISS orders.

        There is a lot of flexibility in this routine, so please read
        the options carefully. There are 2 options for extracting the
        spectra:
        1. Extracting the orders via quadrants. This will remove the
        first and second orders on the righthand side of the image
        (no contamination) first, then extract the overlapping contaminated
        region together.
        2. Extracting the orders by orders. This will extract the
        *entire* first order and the *entire* second order, including
        the overlapping contaminated region for both orders.

        Parameters
        ----------
        proftype : str, optional
           Which type of profile to use to extract the orders.
           Default is `median` or a median-created profile. Other options
           include `gaussian` (which is a Gaussian profile shape) and
           `moffat` (which is a Moffat profile shape).
        sigma : float, optional
           What sigma to look for when removing outliers during the optimal
           extraction routine. Default = 20.
        Q : float, optional
           An estimate on the gain. Default = 1.8.
        per_quad : bool, optional
           Whether to extract the spectra per quadrants (method 1 above)
           or as a full order (method 2 above). Default is `True` (will
           extract spectra via quadrants).
        test : bool, optional
           Whether to run a few test frames or run the entire dataset. Default
           is False (will run all exposures). If `True`, will run the middle 5
           exposures.

        Attributes
        ----------
        opt_order1_flux : np.array
           Optimally extracted flux for the first order.
        opt_order2_flux : np.array
           Optimally extracted flux for the second order.
        opt_order1_err : np.array
           Optimally extracted flux error for the first order.
        opt_order2_err : np.array
           Optimally extracted flux error for the second order.
        """
        if self.box_spectra1 is None:
            self.extract_box_spectrum()

        if self.trace_edge is not None:
            pos1 = self.trace_edge['order_1']
            pos2 = self.trace_edge['order_2']
            pos3 = self.trace_edge['order_3']
        elif self.trace_ear is not None:
            pos1 = self.trace_ear['order_1']
            pos2 = self.trace_ear['order_2']
            pos3 = self.trace_ear['order_3']
        else:
            self.map_trace()
            pos1 = self.trace_edge['order_1']
            pos2 = self.trace_edge['order_2']
            pos3 = self.trace_edge['order_3']

        if test:
            start, end = 0, 5
            # start, end = int(len(self.data)/2-2), int(len(self.data)/2+2)
        else:
            start, end = 0, len(self.data)

        # cr_mask = ~np.array(self.cr_mask, dtype=bool)

        all_fluxes, all_errs, all_profs = optimal_extraction_routine(
            self.data[start:end], self.var[start:end],
            spectrum=np.array([self.box_spectra1[start:end],
                               self.box_spectra2[start:end],
                               self.box_spectra3[start:end]]),
            spectrum_var=np.array([self.box_var1[start:end],
                                   self.box_var2[start:end],
                                   self.box_var3[start:end]]),
            sky_bkg=np.zeros(self.data.shape),  # self.bkg[start:end],
            medframe=self.median,  # cr_mask=self.bkg_removed,
            pos1=pos1, pos2=pos2, pos3=pos3, sigma=sigma, Q=Q,
            proftype=proftype, per_quad=per_quad, isplots=isplots)
        return all_fluxes, all_errs, all_profs
