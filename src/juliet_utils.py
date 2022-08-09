import os
import juliet
import corner
import numpy as np
from tqdm import tqdm
from astropy.table import Table

__all__ = ['create_data_dictionary', 'create_priors_dictionary', 'corner_plot',
           'read_juliet_table', 'compile_posteriors']

def read_juliet_table(filename):
    """
    Takes a csv of priors and reads them into Juliet formatting.

    Parameters
    ----------
    filename : str

    Returns
    -------
    params : np.ndarray
    dists : np.ndarray
    hyperparams : np.ndarray
    """
    priors = Table.read(filename, format='csv')
    params = np.array(priors['parameter'])
    dists  = np.array(priors['dist'])

    hyperparams = []
    for i in range(len(priors)):
        if type(priors['bound'][i]) == np.ma.core.MaskedConstant:
            hyperparams.append(priors['hyperparam'][i])
        else:
            temp = [priors['hyperparam'][i], priors['bound'][i]]
            hyperparams.append(temp)
    return params, dists, hyperparams

def compile_posteriors(output_dir):
    """
    Manuevers into all wave_* directories and picks out p_p1 from each
    posteriors.dat file.

    Parameters
    ----------
    output_dir : str

    Returns
    -------
    wave : np.ndarray
    radii : np.ndarray
    errs : np.ndarray
    """
    paths = np.sort([os.path.join(output_dir, i) for i in
                     os.listdir(output_dir) if i.startswith('wave_')])
    wave = np.zeros(len(paths))
    radii = np.zeros(len(paths))
    errs  = np.zeros(len(paths))

    for i in tqdm(range(len(paths))):
        try:
            tab = Table.read(os.path.join(paths[i], 'posteriors.dat'),
                             format='ascii')
            wave[i] = float(paths[i].split('_')[-1])
            row = tab[tab['col1']=='p_p1']
            radii[i] = row['col2']
            errs[i]  = row['col3']
        except FileNotFoundError:
            pass
    return wave, radii, errs


def create_data_dictionary(centers, wavelength, flux, var, times, binsize=0.1,
                           output_dir=None, key=None):
    """
    Reformats the wavelength, flux, and errors into a Juliet readable dictionary.
    Centers are passed in to create light curves centered on those values with
    binsizes = binsize.

    Parameters
    ----------
    centers : np.ndarray
    wavelength : np.ndarray
    flux : np.ndarray
    var : np.ndarray
    times : np.ndarray
    binsize : float, np.ndarray, optional
       The half width of a given spectroscopic light curve bin. Default is 0.1
       micron.

    Returns
    -------
    data_dictionary : dictionary
    """
    if type(binsize) == float or type(binsize) == int:
        binsize = np.full(len(centers), binsize)

    data_dictionary = {}

    for i in range(len(centers)):
        if len(centers)==1 and key is not None:
            key = key
        else:
            key = 'wave_{}'.format(np.round(centers[i],5))
        q = ((wavelength >= centers[i]-binsize[i]) &
             (wavelength <= centers[i]+binsize[i]) )
        y = np.nansum(flux[:,q], axis=1)
        yerr = np.sqrt(np.nansum(var[:,q]**2, axis=1))

        data_dictionary[key] = {'times':times,
                                'flux':y/np.nanmedian(y),
                                'error':yerr/np.nanmedian(y)}

        # Handles behind-the-scenes creation of certain files
        #   This gets around Juliet breaking a little down the line
        if output_dir:
            try:
                os.mkdir(output_dir)
            except:
                pass
            path = os.path.join(output_dir, key)
        else:
            path = os.path.join('.', key)
        try:
            os.mkdir(path)
        except:
            print("Directory already exists, or failed to be created.")
            pass

        data = np.array([times, y/np.nanmedian(y), yerr/np.nanmedian(y),
                     np.full(len(times), 'SOSS', dtype='U10')]).T

        with open(os.path.join(path, 'lc.dat'), 'w') as f:
            for i in range(len(data)):
                f.write('{0}\n'.format(' '.join(str(e) for e in data[i])))
        with open(os.path.join(path, 'priors.dat'), 'w') as f:
            f.write(' ')

    return data_dictionary

def create_priors_dictionary(keys, params, dists, hyperps):
    """
    Creates the dictionary of priors in Juliet style.

    Parameters
    ----------
    keys : np.ndarray
       A list of keys for the priors dictionary. Need to be the same keys as
       the data dictionary.
    params : np.ndarray
       A list of parameter names given in Juliet style (e.g. `p_p1`).
    dists : np.ndarray
       A list of distribution types for each parameter.
    hyperps : np.ndarray
       A list of hyperparameters to feed in as priors to the Juliet fit.

    Returns
    -------
    priors : dictionary
    """
    if ((len(params) != len(dists)) or (len(dists) != len(hyperps)) or
        (len(params) != len(hyperps))):
        return("Params, dists, and hyperps need to be the same length.")
    else:
        priors = {}

        for i in range(len(keys)):
            temp = {}
            for param, dist, hyperp in zip(params, dists, hyperps):
                temp[param] = {}
                temp[param]['distribution'] = dist
                temp[param]['hyperparameters'] = hyperp
            priors[keys[i]] = temp

        return priors

def corner_plot(results, wavelength_key, posterior_keys=None,
                posterior_names=None):
    """
    Creates cornen plots for a given fit.

    Parameters
    ----------
    results : juliet.fit.fit
       The output of `nestor_transit_fitting.fit_lightcurves`, which is a
       Juliet.fit.fit object.
    wavelength_key : str
       The wavelength key to look at the posterior chains for.
    posterior_keys : np.array, optional
        Array of parameter names to make the corner plot with. If not provided,
        will by default look at the posteriors for all fitted parameters.
    posterior_names : np.array, optional
        Array of x- and y-axes labels for each parameter. Default is None.

    Returns
    -------
    figure
    """

    if posterior_keys is None:
        params = list(results[wavelength_key].posteriors['posterior_samples'].keys())
        params = params[2:]
    else:
        params = posterior_keys

    if posterior_names is None:
        posterior_names = np.copy(params)

    first_time = True

    for i in range(len(params)):
        if first_time:
            posterior_data = results[wavelength_key].posteriors['posterior_samples'][params[i]]
            first_time = False
        else:
            posterior_data  = np.vstack((posterior_data,
                                         results[wavelength_key].posteriors['posterior_samples'][params[i]]))

    posterior_data = posterior_data.T
    figure = corner.corner(posterior_data, labels=params)
    return figure
