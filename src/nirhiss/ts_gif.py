import batman
import os, sys
import matplotlib
import numpy as np
from pylab import *
from astropy import units
import matplotlib.pyplot as plt
from astropy.table import Table, Column
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, Rectangle, Arrow
from mpl_toolkits.axes_grid1 import make_axes_locatable

__all__ = ['get_rainbow', 'gauplot', 'batman_transit', 'setup_gridspec',
           'create_figure']

def get_rainbow(cmap='turbo', N=100):
    """
    Extracts N colors from any matplotlib colormap.

    Parameters
    ----------
    cmap : str, optional
       The name of the matplotlib colormap to use. Default is 'turbo'.
    N : int, optional
       The number of colors to extract from the colormap. Default is 100.

    Returns
    -------
    rainbow : np.array
       Array of HEX colors extracted from the colormap.
    """
    cmap = cm.get_cmap(cmap, N)  # matplotlib color palette name, n colors
    rainbow = []
    for i in range(cmap.N):
        rgb = cmap(i)[:3]  # will return rgba, we take only first 3 so we get rgb
        rainbow.append(matplotlib.colors.rgb2hex(rgb))
    return rainbow

def gauplot(centers, radiuses, ax, xr=None, yr=None):
    """
    Creates a gradient circle to mimic the star.

    Parameters
    ----------
    centers :
    radiuses :
    ax :
    xr :
    yr :
    """
    nx, ny = 1000.,1000.
    xgrid, ygrid = np.mgrid[xr[0]:xr[1]:(xr[1]-xr[0])/nx,yr[0]:yr[1]:(yr[1]-yr[0])/ny]
    im = xgrid*0 + np.nan
    xs = np.array([np.nan])
    ys = np.array([np.nan])
    fis = np.concatenate((np.linspace(-np.pi,np.pi,100), [np.nan]) )
    cmap = plt.cm.OrRd_r
    thresh = 0.1
    for curcen,currad in zip(centers,radiuses):
            curim=(((xgrid-curcen[0])**2+(ygrid-curcen[1])**2)**.5)/currad*thresh
            im[curim<thresh]=np.exp(-.5*curim**2)[curim<thresh]
            xs = np.append(xs, curcen[0] + currad * np.cos(fis))
            ys = np.append(ys, curcen[1] + currad * np.sin(fis))
    return im

def batman_transit(time, Rp, lim=0.15):
    """
    Creates a transit model that only varies the Rp/Rstar parameter.

    Parameters
    ----------
    time : np.array
       The time array to evaluate over.
    Rp : float
       The Rp/Rstar value to create the transit model for.
    lim : float, optional
       The time limits to evaluate over. Default is -0.15 to 0.15 [days].

    Returns
    -------
    time : np.array
       Time array within the time limits.
    flux : np.array
       batman transit model flux.
    """
    time = np.linspace(-lim, lim, len(time))
    params = batman.TransitParams()
    params.t0 = 0                       #time of inferior conjunction
    params.per = 3.4252602                  #orbital period
    params.rp = Rp                       #planet radius (in units of stellar radii)
    params.a = 8.84                       #semi-major axis (in units of stellar radii)
    params.inc = 85.14                    #orbital inclination (in degrees)
    params.ecc = 0.                      #eccentricity
    params.w = 90.                       #longitude of periastron (in degrees)
    params.u = [0.1, 0.3]                #limb darkening coefficients [u1, u2]
    params.limb_dark = "quadratic"       #limb darkening model

    m = batman.TransitModel(params, time)    #initializes model
    flux = m.light_curve(params)          #calculates light curve

    return time, flux

def setup_gridspec():
    # Create the gridspec environment
    fig = plt.figure(tight_layout=True, figsize=(14,8))
    fig.set_facecolor('w')
    gs = GridSpec(2, 3, width_ratios=[1.3,1.3,3.2], height_ratios=[1,0.6])

    text_ax = fig.add_subplot(gs[0,0:2])
    ax = fig.add_subplot(gs[0, 2])
    ax2 = fig.add_subplot(gs[1, 0:])
    return fig, gs, text_ax, ax, ax2

def rmv_axes(ax, text_ax, lim):
    """ Removes the axes from the passed in subplots.
    """
    for a in [ax]:
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
        a.spines['bottom'].set_visible(False)
        a.spines['left'].set_visible(False)
        a.set_yticks([])
        a.set_xticks([])
    for a in [text_ax]:
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
        a.set_xticks([-lim, 0, lim])
        a.set_yticks([0.98, 0.99, 1.0])
    return

def set_limits_labels(ax2, text_ax, t):
    """ Sets the limits and labels for the passed in subplots.
    """
    ax2.set_xlim(0.55, 2.9)
    ax2.set_ylim(2, 2.3)
    ax2.set_xlabel('wavelength [$\mu$m]')
    ax2.set_ylabel('amount of light\nblocked by the\nplanet [%]', fontsize=16)

    text_ax.set_xlabel('time [hours]')

    text_ax.set_xlim(t[0], t[-1]+0.05)
    text_ax.set_xticks([t[0], 0, t[-1]])
    text_ax.set_ylim(0.975, 1.01)
    return

def label_rp(ax):
    ax.hlines(500, 1545, 1690, lw=4, color='pink')
    ax.text(s='$R_{planet}$', x=1500, y=680, color='k')
    return

def label_rs(ax):
    ax.vlines(995, 0, 510, lw=4, color='deepskyblue')
    ax.text(s='$R_{star}$', x=1020, y=280, color='k')
    return

def label_depth(text_ax):
    text_ax.vlines(0, 0.9788, 1, color='xkcd:lilac', lw=3, zorder=5)
    text_ax.hlines(1, -0.005, 0.005, color='xkcd:lilac', lw=3, zorder=5)
    text_ax.hlines(0.9788, -0.005, 0.005, color='xkcd:lilac', lw=3, zorder=5)

    text_ax.text(s='amount of\nlight\nblocked\nby planet = ',
                 x=-0.04, y=0.992, fontsize=14)

    text_ax.text(s=r'$\left( \frac{R_{planet}}{R_{star}} \right)^2$',
                 x=0.002, y=0.992,
                 fontsize=22)
    return text_ax

def label_spec(ax2, wave, dppm):
    ax2.text(s=r'$\left( \frac{R_{planet}}{R_{star}} \right)$',
             x=wave+0.03,
             y=dppm-0.0003,
             fontsize=24)
    return

def create_planet(x, tstart, i, move, star, padding, dppm, ax):
    # Adds the planet
    rprs = np.sqrt(dppm/1e6)
    planet = Circle((x+tstart+i+move[i], (star.shape[0]-padding*2)/2),
                    np.nanmedian(rprs)*500,
                    color='k')
    ax.add_patch(planet)
    return

def create_atm(x, tstart, i, move, star, padding, dppm, rainbow, color_ind, ax):
    # Adds the atmosphere
    rprs = np.sqrt(dppm/1e6)
    atm = Circle((x+tstart+i+move[i], (star.shape[0]-padding*2)/2),
                 1000*rprs[color_ind],
                 color=rainbow[color_ind],
                 alpha=0.4)
    ax.add_patch(atm)
    return

def add_time_wavelength_text(text_ax, t, i, lim, wavelength, color_ind):
    tt = ((t[i]+lim)*units.day).to(units.hour).value
    text_ax.text(s='time = {:.3f} hours'.format(np.round(tt,3)), x=t[50],
                 y=1.009, fontsize=16)
    text_ax.text(s='wavelength = {:.3f} $\mu$m'.format(np.round(wavelength[color_ind],
                                                                3)),
                 x=t[50], y=1.006, fontsize=16)
    return

def create_figure(wavelength, dppm, dppm_err, outputdir, loop='time',
                  star_cmap='OrRd_r', labelrp=False, labelrs=False,
                  labeldepth=False, labelspec=False,
                  dontswap=True, transitbehind=False):
    """
    Creates the transit spectroscopy figure.

    Parameters
    ----------
    wavelength : np.array
       Center wavelengths for the transmission spectrum.
    dppm : np.array
       Measured dppm for the transmission spectrum.
    dppm_err : np.array
       Measured errors on the dppm for the transmission spectrum.
    loop : str, optional
       What parameter to loop over per each frame. Default is 'time'. Other
       option is 'wavelength'.
    star_cmap : str, optional
       The name of the matplotlib colormap to use for the stellar disk. Default
       is 'OrRd_r'.
    labelrp : bool, optional
       Label the variable Rp on the planet transiting subplot. Default is False.
    labelrs : bool, optional
       Label the variable Rstar on the planet transiting subplot. Default is
       False.
    labeldepth : bool, optional
       Label the variable (Rp/Rs)^2 on the batman transit model subplot. Default
       is False.
    labelspec : bool, optional
       Label the variable (Rp/Rs) on the transmission spectrum plot. Default is
       False.
    """
    np.random.seed(456)

    x = 5
    lim = 0.061

    time = np.arange(0,900,1)
    move = np.linspace(5,620,len(time))

    tstart=200
    tend=1050

    rainbow = get_rainbow(N=len(dppm))
    rprs = np.sqrt(dppm/1e6)
    t, transit_model = batman_transit(time, rprs[0],
                                      lim=lim)
    transit_model_noise = np.random.normal(0,0.0003,len(transit_model))

    if loop == 'time':
        color_ind = 0

        nframes_per = int(len(t)/len(dppm))

        for i in range(len(time)):

            fig, gs, text_ax, ax, ax2 = setup_gridspec()

            if transitbehind == True:
                addition = '_behind'
            else:
                addition = '_front'

            star = gauplot([(x,x)], [x], ax, [x-5, x+5], [x-5, x+5])
            padding = 500
            star = np.pad(star, (padding,padding), mode='constant',
                          constant_values=np.nan)

            # Resets the color of the atmosphere to the last one
            if dontswap == False:
                if i > 470:
                    color_ind = -1
                    t, transit_model = batman_transit(time,
                                                      rprs[color_ind],
                                                      lim=lim)
            if i == 880:
                if labelrp == True:   # Labels Rp on the subplot ax
                    label_rp(ax)
                    addition = '_front_a'
                    print('good a')
                if labelrs == True:  # Labels Rp and Rs on the subplot ax
                    label_rp(ax)
                    label_rs(ax)
                    addition = '_front_b'
                    print('good b')
                if labeldepth == True: # Labels the depth on text_ax and parameters on ax
                    label_depth(text_ax)
                    label_rp(ax)
                    label_rs(ax)
                    addition = '_front_c'
                    print('good c')
                if labelspec == True:  # Labels all components
                    label_spec(ax2, wavelength[color_ind],
                               dppm[color_ind])
                    label_depth(text_ax)
                    label_rp(ax)
                    label_rs(ax)
                    ax2.errorbar(wavelength[0], dppm[0]/1e4,
                                 marker='o',
                                 color=rainbow[0], ms=8, lw=3, linestyle='',
                                 yerr=dppm_err[0]/1e4)
                    addition = '_front_d'

            if i > 470 and dontswap == False:
                ax2.errorbar(wavelength, dppm/1e4, marker='o',
                             color='#404040', ms=5, linestyle='',
                             yerr=dppm_err/1e4)

            # Adds the star
            if transitbehind == True:
                create_atm(x, tstart, i, move, star, padding, dppm, rainbow,
                           color_ind, ax)
                create_planet(x, tstart, i, move, star, padding, dppm, ax)
                ax.imshow(star[padding:-padding], cmap=star_cmap)

                text_ax.plot(t, transit_model+transit_model_noise,
                             '.', color=rainbow[color_ind])

            else:
                ax.imshow(star[padding:-padding], cmap=star_cmap)
                create_atm(x, tstart, i, move, star, padding, dppm, rainbow,
                           color_ind, ax)
                create_planet(x, tstart, i, move, star, padding, dppm, ax)

                text_ax.plot(t[:i], transit_model[:i]+transit_model_noise[:i],
                             '.', color=rainbow[color_ind])

            # Adding text
            add_time_wavelength_text(text_ax, t, i, lim, wavelength, color_ind)
            set_limits_labels(ax2, text_ax, t)
            rmv_axes(ax, text_ax, lim)

            plt.savefig(os.path.join(outputdir, 'figure_{0:04d}{1}.png'.format(i, addition)),
                        dpi=150, rasterize=True, bbox_inches='tight')
            plt.close()


    elif loop == 'wavelength':
        i = 880

        xoffset = np.arange(0, 0.08, 0.0006)
        yoffset = np.arange(0, 0.01, 0.00004)

        # Creates transits for all Rp/Rs values
        transit_grid = np.zeros((len(dppm), len(time)))
        for r in range(len(dppm)):
            t, transit_model = batman_transit(time, rprs[r], lim=lim)
            transit_grid[r] = transit_model + transit_model_noise

        for color_ind in range(len(dppm)):
            fig, gs, text_ax, ax, ax2 = setup_gridspec()

            star = gauplot([(x,x)], [x], ax, [x-5, x+5], [x-5, x+5])
            padding = 500
            star = np.pad(star, (padding,padding), mode='constant',
                          constant_values=np.nan)
            ax.imshow(star[padding:-padding], cmap=star_cmap)

            # Adds the atmosphere
            create_atm(x, tstart, i, move, star, padding, dppm, rainbow,
                       color_ind, ax)
            create_planet(x, tstart, i, move, star, padding, dppm, ax)

            subsetx = np.flip(xoffset[:color_ind])
            subsety = np.flip(yoffset[:color_ind])
            # Updating the transit model with the correct dppm & plots
            for c in range(color_ind-1):
                text_ax.plot(t+subsetx[c],
                             transit_grid[c]+subsety[c],
                             marker='.', color=rainbow[c], lw=0,
                             markeredgecolor='none', alpha=0.2)

            text_ax.plot(t, transit_grid[color_ind],
                         marker='.', ms=10,
                         markeredgecolor=rainbow[color_ind],
                         color='w', lw=0.1)

            # Plots the transmission spectrum
            ax2.errorbar(wavelength[:color_ind], dppm[:color_ind]/1e4,
                         marker='o', color='#404040', ms=5, linestyle='',
                         yerr=dppm_err[:color_ind]/1e4)

            ax2.errorbar(wavelength[color_ind], dppm[color_ind]/1e4,
                         marker='o',
                         yerr=dppm_err[color_ind]/1e4,
                         color=rainbow[color_ind], ms=8,
                         lw=3)

            # Adding text
            add_time_wavelength_text(text_ax, t, i, lim, wavelength, color_ind)
            set_limits_labels(ax2, text_ax, t)
            rmv_axes(ax, text_ax, lim)
            plt.savefig(os.path.join(outputdir,
                                     'figure_{0:04d}d_{1:03d}.png'.format(i, color_ind+1)),
                        dpi=150, rasterize=True, bbox_inches='tight')
            plt.close()
