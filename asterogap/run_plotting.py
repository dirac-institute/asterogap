import argparse
import textwrap

import numpy as np
import corner
import matplotlib.pyplot as plt
import h5py
from asterogap.GP import GPFit


def calc_periods(data, nperiods=1, period=None, p_range=None, bins=20, width=0.1):
    """
    NOTE: Should work for both single-kernel and double-kernel results
    Roughly calculates the most likely period(s).

    Parameters
    ----------
    data : numpy.ndarray
        Results pulled from hdf5 file. Assumes the shape to be [nwalkers, iterations, parameters].

    nperiods : int
        The number of top periods to look for.

    period : float
        The period (in hours) around which to calculate the probability.

    p_range : list
        List of starting and ending values for an interval.

    bins : int
        The number of bins to calculate with for the numpy histogram. It's best to keep this number very large.

    width : float
        The distance on either side of the period over which will be integrated. Width will be ignored
        if p_range is specified.

    Returns
    -------

    probs : numpy.float64
        Total integrated area (probability) of the posterior period distribution within the edges listed.

    edges : list
        List of starting and ending values for the interval used to calculate the probability.
        This will be the same as p_range if p_range was specified.

    """


    flat_data = data.reshape(data.shape[0] * data.shape[1], data.shape[2])

    h, bins = np.histogram(flat_data[:, -1], bins=bins, density=True)

    # find the period(s) of hmax
    top_h = -np.sort(-h)[0:nperiods]

    # collect the indicies for these periods
    indices = []
    for i, v in enumerate(top_h):
        ind = np.where(h == top_h[i])
        indices.append(ind[0][0])

    indices = np.array(indices)

    # find the width of the bins
    dx = bins[1] - bins[0]

    periods = bins[indices] + dx/2. # add half the bin to center
    #
    # edges = [periods - width/2., periods + width/2.]
    #
    # prob_sum = []
    #
    # for i, v in enumerate(edges[0]):
    #     # find all the periods that fall within the defined edges
    #     periods_where = flat_data[:,-1][np.where((flat_data[:,-1]>= edges[0][i]) & (flat_data[:,-1] <= edges[1][i]))]
    #
    #     # find the sum of the number of walkers within the edges and add to list
    #     prob_sum.append(periods_where.shape[0]/flat_data.shape[0])
    #
    # return prob_sum, edges
    return periods



def plot_corner(data, true_period=None, colours=None, zoom=False, trim=None, fig=None):
    """
    NOTE: Should work for both single-kernel and double-kernel results.
    Plot a corner plot showing the projections of a data set in multi-dimesional space,
    with the different dimensions corresponding to the different kernel parameters.

    Parameters
    ----------
    data : numpy.ndarray
        Results pulled from hdf5 file. Assumes the shape to be [nwalkers, iterations, parameters].

    true_period : float
        The true period of the asteroid light curves.

    colours : [str, str, str]
        List of (up to) three colours. First colour is used for the data, the second
        colour for the true underlying data, the third for the models.
        Note, for a corner plot, the first colour will always default to black.
        Only the second true underlying data color can be changed.

    zoom : bool
        Toggles whether the corner plot will show a zoomed in version of the histogram,
        focusing on the densest region of the previous binned histogram.

    trim : [float, float]
        Specifies what percentile range of the period is to be used for plotting.
        This can be used to eliminate extreme outliers that would otherwise distort the plot.

    Returns
    -------

    figure : matplotlib.figure.Figure
        The object with the plot

    """

    if colours is None:
        colours = ["#000000", "#0072B2", "#E69F00", "#009E73", "#F0E442"]

    if trim:

        try:
            lower, upper = np.percentile(data[:, :, -1], [trim[0], trim[1]])
            flat_data = data[(data[:, :, -1] > lower) & (data[:, :, -1] < upper)]

        except TypeError:
            print(
                "ERROR: Values for trim must be formatted as an array (e.g. [5], not 5).\n Plot will be processed without any trimming."
            )
            flat_data = data.reshape(data.shape[0] * data.shape[1], data.shape[2])

        except IndexError:
            print(
                "WARNING: No upper percentile was specified. Upper percentile was set to 100."
            )
            trim = np.append(trim, 100)
            lower, upper = np.percentile(data[:, :, -1], [[0], trim[1]])

            flat_data = data[(data[:, :, -1] > lower) & (data[:, :, -1] < upper)]

    else:
        flat_data = data.reshape(data.shape[0] * data.shape[1], data.shape[2])

    if zoom:
        periods = calc_periods(data, period=true_period, )

        width = 0.5
        edges = [periods - width/2., periods + width/2.]

        # if np.any(prob == 0):
        #     raise Exception(
        #         "WARNING: Probability around period is 0 and therefore cannot display a valid corner plot."
        #     )

        flat_data = data[(data[:, :, -1] > edges[0]) & (data[:, :, -1] < edges[1])]

    if data.shape[2] == 6:
        labels = ["mean", "log_amp_long", "log_metric", "log_amp", "gamma", "period"]
        truths = [None, None, None, None, None, true_period]

    else:
        labels = ["mean", "log_amp", "gamma", "period"]
        truths = [None, None, None, true_period]

    figure = corner.corner(
        flat_data,
        labels=labels,
        truths=truths,
        show_titles=True,
        title_kwargs={"fontsize": 9},
    )

    return figure


def plot_trace(data, iterations, colours=None):
    """
    NOTE: Should work for both single-kernel and double-kernel results
    Create a plot of walker movement after the burn-in period.

    Parameters
    ----------
    data : numpy.ndarray
        Results pulled from hdf5 file. Assumes the shape to be [nwalkers, iterations, parameters].

    iterations : int
        The number of iterations the MCMC code was run for. Can use data.shape[1].

    colours : [str]
        One colour. First colour is used for the data.

    Returns
    -------

    fig : matplotlib.figure.Figure
        The object with the plot

    """

    if colours is None:
        colours = ["black"]

    # print(int(data.shape[2]/2))

    fig, ax = plt.subplots(
        2, int(data.shape[2] / 2), figsize=(5 * int(data.shape[2] / 2), 9)
    )
    fig.subplots_adjust(wspace=0.5, hspace=0.3)

    if data.shape[2] == 6:
        dims = ["mean", "log_amp_long", "log_metric", "log_amp_periodic", "log_gamma", "period"]
        axs = [ax[0, 0], ax[0, 1], ax[0, 2], ax[1, 0], ax[1, 1], ax[1, 2]]

    else:
        dims = ["mean", "log_amp", "log_gamma", "period"]
        axs = [ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1]]

    x = np.arange(iterations)

    for i in range(data.shape[2]):
        axs[i].set_xlabel("Step Number")
        axs[i].set_ylabel("{}".format(dims[i]))

        for j in range(data.shape[0]):
            param = data[j, :, i]
            axs[i].plot(x, param, color=colours[0], alpha=0.3)

    return fig


def run_lsp(
    time,
    flux,
    flux_err=None,
    data=None,
    true_period=None,
    true_lightcurve=None,
    use_radians=False,
    legend=True,
    colours=None,
    plot=False,
    ):
    """
    NOTE: Should work for both single-kernel and double-kernel results
    Determine the Lomb-Scargle Periodogram (nterms = 1-3) for the light curve data
    and plot the frequencies as well as the a folded light curve using

    Parameters
    ----------
    time : numpy.ndarray
        The time stamps of the periodic light curve

    flux : numpy.ndarray
        Flux measurements corresponding to the time stamps

    flux_err : numpy.ndarray
        The flux uncertainties corresponding to the data.

    data : numpy.ndarray
        Results pulled from hdf5 file. Assumes the shape to be [nwalkers, iterations, parameters].

    true_period : float
        The period on which to fold in hours

    true_lightcurve : iterable containing (true_time, true_flux)
        In the case of simulated data, this contains the times and flux values from which the
        simulated data was created (could be higher-resolution than the "data"), useful for
        comparison between models created e.g. from MCMC samples and the true underlying process

    use_radians : bool, default False
        If True, the phase will be plotted from (0, 2pi) instead of (0,1), which is the default.

    legend : bool, default True
        If True, include a legend in the plot

    colours : [str, str, str]
        List of (up to) three colours. First colour is used for the data, the second
        colour for the true underlying data, the third for the models.

    plot : bool
        Do you want a plot?

    Returns
    -------

    figure : matplotlib.figure.Figure
        The object with the plot

    lsp_period : numpy.ndarray
        An array of the periods calculated from the Lomb-Scargle periodograms with corresponding
        n_terms [1,2,3].
    """

    if colours is None:
        colours = ["#000000", "#0072B2", "#E69F00", "#009E73", "#F0E442"]

    # NEW LSP CODE
    #---------------------

    from gatspy import periodic

    if data is not None:
        lower, upper = np.percentile(data[:, :, -1], [5, 95])
        masked_data = data[(data[:, :, -1] > lower) & (data[:, :, -1] < upper)]


    # set up 2term LSP model
    model = periodic.LombScargle(Nterms=2)

    max_period=60.0/24.0 # 2.5 days
    # make sure the max period isn't longer than the time range of observations
    big_period = np.min([max_period, (time.max() - time.min())])
    min_period=1.0/24.0 # 1 hour

    model.optimizer.period_range = (min_period, big_period)
    model.optimizer.first_pass_coverage = 200

    # add the obs data
    model.fit(time, flux, flux_err)

    # find the 5 best periods (in days)
    best_periods = model.find_best_periods()

    if plot:
        fig, ax = plt.subplots(len(best_periods)+1, figsize=(10, 4*(len(best_periods)+1)))

        # have range spanning the min period-1 hour (unless that's less than 0) to max period+1 hour
        p_range = np.linspace(np.max([0, best_periods.min()-1/24.]), best_periods.max()+1/24., 1000)

        # calculate LSP for the range
        LSP = model.periodogram(p_range)

        ax[0].plot(p_range*24., LSP, color=colours[0], alpha=0.7)

        # set xlim same as p_range
        ax[0].set_xlim([np.max([0, best_periods.min()-1/24.])*24, best_periods.max()*24.+1])

        if data is not None:
            #lower, upper = np.percentile(data[:, :, -1], [5, 95])
            lower = p_range.min()*24.
            upper = p_range.max()*24.
            masked_data = data[(data[:, :, -1] > lower) & (data[:, :, -1] < upper)]


        if data is not None:
            ax[0].hist(
                masked_data[:, -1],
                bins=20,
                color=colours[3],
                alpha=0.5,
                density=True,
                label="Posterior",
            )

        ax[0].set_xlabel("Period (hrs)")
        ax[0].set_ylabel("Normalized Power")
        ax[0].set_ylim([0, 1])

        if legend:
            ax[0].legend()

        for i in np.arange(len(best_periods)):

            ax[0].vlines(
                best_periods[i]*24.,
                0,
                1,
                colors=colours[2],
                linestyles="--",
                #label="Best fit : %.5f" % (best_periods[i] * 24.0),
            )

            plot_folded_lightcurve(
                time,
                flux,
                period=best_periods[i] * 24.0,
                ax=ax[i+1],
                true_lightcurve=true_lightcurve,
                use_radians=use_radians,
                legend=False,
            )

        plt.tight_layout()


    #---------------------

    #
    # from scipy.signal import argrelextrema
    #
    # # get l-s best period estimate
    # from lombscargle import make_lsp
    # # from astropy.timeseries import LombScargle
    #
    # if data is not None:
    #     lower, upper = np.percentile(data[:, :, -1], [5, 95])
    #     masked_data = data[(data[:, :, -1] > lower) & (data[:, :, -1] < upper)]
    #
    # lsp_periods = np.array([])
    #
    # if plot:
    #     fig, ax = plt.subplots(3, 2, figsize=(10, 10), squeeze=False)
    #
    # for i in np.arange(3):
    #     freq, power = make_lsp(time, flux, flux_err, p_max=5.0, nterms=(i + 1))
    #
    #     # determine the indices of local power maxima
    #     best_idx = argrelextrema(power, np.greater)
    #
    #     # sort these indices based on actual power value
    #     # reverse list so max is read first
    #     indices = np.argsort(power[best_idx[0]])[::-1]
    #
    #     # sort our original indices based on the new
    #     # power-sorted indices
    #     best_idx = (best_idx[0]).T[indices]
    #     best_freqs = freq[best_idx].T
    #
    #     new_freq = best_freqs[0]
    #     new_period = 1.0 / new_freq
    #
    #     if plot:
    #
    #         # plot all the frequencies
    #         ax[i][0].plot((1.0 / freq) * 24.0, power, color=colours[0], alpha=0.7)
    #
    #         if data is not None:
    #             ax[i][0].hist(
    #                 masked_data[:, -1],
    #                 bins=20,
    #                 color=colours[3],
    #                 alpha=0.5,
    #                 density=True,
    #                 label="Posterior",
    #             )
    #
    #         y_max = (ax[i][0].get_ylim())[1]
    #
    #         ax[i][0].vlines(
    #             new_period * 24.0,
    #             0,
    #             y_max,
    #             colors=colours[2],
    #             linestyles="--",
    #             label="Best fit : %.5f" % (new_period * 24.0),
    #         )
    #
    #         if true_period:
    #             ax[i][0].vlines(
    #                 true_period,
    #                 0,
    #                 y_max,
    #                 colors=colours[1],
    #                 linestyles="--",
    #                 label="True fit : %.5f" % true_period,
    #             )
    #
    #         ax[i][0].set_xlabel("Period (hrs)")
    #         ax[i][0].set_ylabel("Normalized Power")
    #         ax[i][0].set_title("nterms = %s" % (i + 1))
    #         ax[i][0].set_xlim([0, 24])
    #         ax[i][0].set_ylim([0, y_max])
    #
    #         if legend:
    #             ax[i][0].legend()
    #
    #         plot_folded_lightcurve(
    #             time,
    #             flux,
    #             period=new_period * 24.0,
    #             ax=ax[i][1],
    #             true_lightcurve=true_lightcurve,
    #             use_radians=use_radians,
    #             legend=False,
    #         )
    #
    #     lsp_periods = np.append(lsp_periods, new_period * 24.0)
    #
    # if plot:
    #     plt.tight_layout()

    return best_periods


def plot_posterior(data, true_period=None, legend=True, colours=None):
    """
    NOTE: Should work for both single-kernel and double-kernel results
    Plot a histogram of the posterior distribution, showing the full distribution,
    the 5th-95th percentile of the distribution, and a zoomed-in view of
    the region with the highest probability (or region around the period if specified).

    Parameters
    ----------
    data : numpy.ndarray
        Results pulled from hdf5 file. Assumes the shape to be [nwalkers, iterations, parameters].

    true_period : float
        The period (in hours) around which to calculate the probability.

    legend : bool, default True
        If True, include a legend in the plot

    colours : [str, str, str]
        List of (up to) three colours. First colour is used for the data, the second
        colour for the true underlying data, the third for the models.

    Returns
    -------

    ax : matplotlib.Axes object
        The object with the plot


    """

    if colours is None:
        colours = ["black", "#0072B2", "#E69F00", "#009E73", "#F0E442"]

    fig, ax = plt.subplots(3, 2, figsize=(10, 15))

    # plot the full histogram of period results
    flat_data = data.reshape(data.shape[0] * data.shape[1], data.shape[2])

    ax[0, 0].hist(flat_data[:, -1], bins="scott", density=True, color=colours[0], alpha=0.3)

    if true_period:
        ylim = ax[0, 0].get_ylim()
        ax[0, 0].vlines(
            x=true_period,
            ymin=ylim[0],
            ymax=ylim[-1],
            lw=1,
            color=colours[1],
            linestyle="dashed",
            label="true period : %.5f" % true_period,
        )

    ax[0, 0].set_xlabel("Period in hours")
    ax[0, 0].set_ylabel("Probability")
    ax[0, 0].set_ylim(ax[0, 0].get_ylim())
    ax[0, 0].set_title("Posterior Period Distibution")

    # plot the 5th-95th percentile
    lower, upper = np.percentile(data[:, :, -1], [5, 95])
    masked_data = data[(data[:, :, -1] > lower) & (data[:, :, -1] < upper)]

    ax[0, 1].hist(masked_data[:, -1], bins="scott", density=True, color=colours[0], alpha=0.3)

    if true_period:
        ylim = ax[0, 1].get_ylim()
        ax[0, 1].vlines(
            x=true_period,
            ymin=ylim[0],
            ymax=ylim[-1],
            lw=1,
            color=colours[1],
            linestyle="dashed",
            label="true period : %.5f" % true_period,
        )

    ax[0, 1].set_title("5th - 95th Percentile")
    ax[0, 1].set_xlabel("Period in hours")
    ax[0, 1].set_ylabel("Probability")
    ax[0, 1].set_ylim(ax[0, 1].get_ylim())

    # zoom in on the part of the graph that has the highest probability
    periods = calc_periods(data, 4, true_period)
    #
    # if not np.any(probs):
    #     raise Exception(
    #         "WARNING: Probability around period is 0 and therefore cannot display a valid corner plot."
    #     )

    best_periods = []
    probs = []

    for i, p in enumerate(periods):
        # trim data to +/- 10-20% of what the period is
        # dividing the log2 of the period by 35 should give you
        # ~10% for 10 hours, ~20% for 200 hours, and ~30% for 1500 hours
        # and if the period is less than 10 hour (since the log2(1)=0),
        # then just set the trim to 10%
        if p > 10:
            period_data = flat_data[(flat_data[:,-1]>(p-p*np.log2(p)/35)) & (flat_data[:,-1]<(p+p*np.log2(p)/35)) ]

        else:
            period_data = flat_data[(flat_data[:,-1]>(p-p*0.1)) & (flat_data[:,-1]<(p+p*0.1)) ]

            if len(period_data)==0:
                # this can happen if the period is too close to 0
                period_data = flat_data[(flat_data[:,-1]>0) & (flat_data[:,-1]<(p+p*0.1)) ]

        h, bins = np.histogram(period_data[:,-1], bins=1000, density=True)

        top_h = -np.sort(-h)[0:1]
        half_h = top_h/2

        # half max h values
        hmin = h[h>half_h][0]
        hmax = h[h>half_h][-1]

        # if the half_h is too much (aka hmin is the same as hmax), try 10%?
        # this can happen if curve is too steep
        if hmax==hmin:
            hmin = h[h>(0.1*top_h)][0]
            hmax = h[h>(0.1*top_h)][-1]

        bin_edges = [bins[np.where(h==hmin)[0][0]], bins[np.where(h==hmax)[0][0]]]

        bin_center = bins[np.where(h==top_h)]

        # move the bin edges to 3 sigma away
        bin_edges[0] = bin_center - 3*(bin_center - bin_edges[0])
        bin_edges[1] = bin_center + 3*(bin_edges[1] - bin_center)

        # if the 10%h fix didn't work earlier, manually move the bins out one each,
        # so that there's at least something to plot
        if bin_edges[0] == bin_edges[1]:
            bin_edges[0] = bins[np.where(h==hmin)[0][0]-1]
            bin_edges[1] = bins[np.where(h==hmin)[0][0]+1]

        # now zoom in on the half-width half-max * 3 area
        zoom_data = data[(data[:, :, -1] > bin_edges[0]) & (data[:, :, -1] < bin_edges[1])]

        #now we can see what the probs are
        prob = period_data.shape[0]/(data.shape[0] * data.shape[1])
        probs = np.append(probs, prob)

        # plot and get bin info
        # set density to false so the y-axis gives a relative sense of scale
        # for the different periods
        n, bins, p = ax[1+int(i/2), i % 2].hist(zoom_data[:, -1], bins="auto", density=False, color=colours[0], alpha=0.3, label="probability: %.3f"%prob)

        # find the bin with the max n and add half a bin width
        best_period = bins[np.where(n==n.max())]+(bins[1]-bins[0])/2
        best_periods = np.append(best_periods, best_period)


        ylim = ax[1+int(i/2), i % 2].get_ylim()
        xlim = ax[1+int(i/2), i % 2].get_xlim()

        if true_period:
            ax[1+int(i/2), i % 2].vlines(
                true_period,
                0,
                ylim[-1],
                lw=1,
                color=colours[1],
                linestyle="dashed",
                label="true period : %.5f" % true_period,
            )
            ax[1+int(i/2), i % 2].vlines(
                best_period,
                0,
                ylim[-1],
                lw=1,
                color=colours[2],
                linestyle="dashed",
                label="best period : %.5f" % best_period,
            )
        else:
            ax[1+int(i/2), i % 2].vlines(
                best_period,
                0,
                ylim[-1],
                lw=1,
                color=colours[2],
                linestyle="dashed",
                label="best period : %.5f" % best_period,
            )
        #ax[1+int(i/2), i % 2].set_title("Probability %.3f" % probs[i])
        ax[1+int(i/2), i % 2].set_xlabel("Period in hours")
        ax[1+int(i/2), i % 2].set_ylabel("Probability")
        ax[1+int(i/2), i % 2].set_ylim(ylim)
        ax[1+int(i/2), i % 2].set_xlim(xlim)

        if legend:
            if true_period:
                ax[0, 0].legend()
                ax[0, 1].legend()
            ax[1+int(i/2), i % 2].legend()

    plt.tight_layout()

    return best_periods, probs


def plot_folded_lightcurve(
    time,
    flux,
    period,
    flux_err=None,
    models=None,
    true_lightcurve=None,
    ax=None,
    use_radians=False,
    legend=True,
    colours=None,
):
    """
    NOTE: Should work for both single-kernel and double-kernel results
    Plot a folded periodic light curve, potentially including the true underlying
    model that produced the data (in the case of simulations), or model
    light curves from MCMC.

    Parameters
    ----------
    time : numpy.ndarray
        The time stamps of the periodic light curve

    flux : numpy.ndarray
        Flux measurements corresponding to the time stamps

    flux_err : numpy.ndarray
        The flux uncertainties corresponding to the data.

    period : array
        The period on which to fold **in hours**

    models : iterable of shape (model_time, numpy.ndarray of shape (nsamples, len(model_time)))
        First element here contains the time stamps for the models (which may not be the same
        as for the data), the second is an array of shape (nsamples, ndatapoints), where nsamples
        is the number of model light curves, and ndatapoints == len(model_time)

    true_lightcurve : iterable containing (true_time, true_flux)
        In the case of simulated data, this contains the times and flux values from which the
        simulated data was created (could be higher-resolution than the "data"), useful for
        comparison between models created e.g. from MCMC samples and the true underlying process


    ax : matplotlib.Axes object
        An Axes object in which to plot the results. If not given, the code will create a
        new figure.

    use_radians : bool, default False
        If True, the phase will be plotted from (0, 2pi) instead of (0,1), which is the default.

    legend : bool, default True
        If True, include a legend in the plot

    colours : [str, str, str]
        List of (up to) three colours. First colour is used for the data, the second
        colour for the true underlying data, the third for the models.

    Returns
    -------

    ax : matplotlib.Axes object
        The object with the plot

    """

    if colours is None:
        colours = ["#000000", "#0072B2", "#E69F00", "#009E73", "#F0E442"]

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    period_days = period / 24.0

    t0 = np.min(time)

    if models:
        t0 = np.min([t0, np.min(models[0])])

    if true_lightcurve:
        t0 = np.min([t0, np.min(true_lightcurve[0])])

    phase = (time - t0) / period_days - np.floor((time - t0) / period_days)

    if use_radians:
        phase *= 2.0 * np.pi

    if flux_err is None:
        ax.scatter(phase, flux, s=5, color=colours[0], label="Observations")
    else:
        ax.errorbar(
            phase,
            flux,
            yerr=flux_err,
            fmt="o",
            c=colours[0],
            markersize=5,
            label="Observations",
        )

    if true_lightcurve:
        true_time = true_lightcurve[0] - t0
        true_flux = true_lightcurve[1]
        true_phase = true_time / period_days - np.floor(true_time / period_days)

        if use_radians:
            true_phase *= 2.0 * np.pi

        # compute the difference from one phase bin to the next
        tdiff = np.diff(true_phase)
        # find all differences < 0, which is where the phase wraps around
        idx = np.where(tdiff < 0)[0]

        # loop through indices where phase goes from 1 (or 2pi) to 0
        # plot each phase light curve separately
        istart = 0
        iend = idx[0] + 1

        # first phase cycle also contains the label for the legend
        ax.plot(
            true_phase[istart:iend],
            true_flux[istart:iend],
            alpha=0.3,
            c=colours[1],
            label="True Lightcurve",
        )

        for i, x in enumerate(idx[:-1]):
            ax.plot(
                true_phase[istart:iend],
                true_flux[istart:iend],
                alpha=0.3,
                c=colours[1],
                label="",
            )
            istart = x + 1
            iend = idx[i + 1] + 1

        # last plot
        istart = idx[-1] + 1
        ax.plot(
            true_phase[istart:], true_flux[istart:], alpha=0.3, c=colours[1], label=""
        )

    # TODO: figure out how to actually use this code
    if models:

        m_time = models[0] - t0
        m_flux = models[1]

        m_phase = (m_time / period_days) - np.floor(m_time / period_days)
        #print("mphase " + str(m_phase))
        if use_radians:
            m_phase *= 2.0 * np.pi

        # compute the difference from one phase bin to the next
        tdiff = np.diff(m_phase)

        #print("tdiff " + str(tdiff))
        # find all differences < 0, which is where the phase wraps around
        idx = np.where(tdiff < 0)[0]
        # if idx.size == 0:
        #    idx = np.array(0)

        # loop through the different samples
        for i, m in enumerate(m_flux):
            # loop through indices where phase goes from 1 (or 2pi) to 0
            # plot each phase light curve separately
            istart = 0
            iend = idx[0] + 1

            if i == 0:
                # first phase cycle also contains the label for the legend
                ax.plot(
                    m_phase[istart:iend],
                    m[istart:iend],
                    alpha=0.1,
                    c=colours[2],
                    label="model",
                )

            else:
                ax.plot(
                    m_phase[istart:iend],
                    m[istart:iend],
                    alpha=0.1,
                    c=colours[2],
                    label="",
                )

            form_time = models[0] - t0
        m_flux = models[1]

        m_phase = (m_time / period_days) - np.floor(m_time / period_days)
        if use_radians:
            m_phase *= 2.0 * np.pi

        # compute the difference from one phase bin to the next
        tdiff = np.diff(m_phase)

        #print("tdiff " + str(tdiff))
        # find all differences < 0, which is where the phase wraps around
        idx = np.where(tdiff < 0)[0]
        # if idx.size == 0:
        #    idx = np.array(0)

        # loop through the different samples
        for i, m in enumerate(m_flux):
            # loop through indices where phase goes from 1 (or 2pi) to 0
            # plot each phase light curve separately
            istart = 0
            iend = idx[0] + 1

            if i == 0:
                # first phase cycle also contains the label for the legend
                ax.plot(
                    m_phase[istart:iend],
                    m[istart:iend],
                    alpha=0.1,
                    c=colours[2],
                    label="model",
                )

            else:
                ax.plot(
                    m_phase[istart:iend],
                    m[istart:iend],
                    alpha=0.1,
                    c=colours[2],
                    label="",
                )

            for j, x in enumerate(idx[:-1]):
                ax.plot(
                    m_phase[istart:iend],
                    m[istart:iend],
                    alpha=0.1,
                    c=colours[2],
                    label="",
                )

                istart = x + 1
                iend = idx[j + 1] + 1

            # last plot
            istart = idx[-1] + 1
            ax.plot(m_phase[istart:], m[istart:], alpha=0.1, c=colours[2], label="")

    if legend:
        ax.legend()
    ax.set_xlabel("Rotational Phase")
    ax.set_ylabel("Flux")
    ax.set_title(r"period $P = %.5f hours$" % period)
    if use_radians:
        ax.set_xlim(0, 2 * np.pi)
    else:
        ax.set_xlim(0, 1)
    plt.tight_layout()

    return ax

def read_data(filename, period_set=None):
    """
    Function reading in the resultant HDF5 file produced from run_gp.py.

    Parameters
    ----------
    filename : string
        Name of the HDF5 file to be read in

    period_set: int
        If the true period is known and should be included in some of the plots

    Returns
    ----------
    data : MCMC chains

    time : original time data used for MCMC run

    flux : original flux data used for MCMC run

    flux_err : original flux error data used for MCMC run

    true_period : the true period that was set, or None if it wasn't


    """
    with h5py.File(filename, "r") as f:
        data = f["chain"][:]
        time = f["time"][:]
        flux = f["flux"][:]
        flux_err = f["flux_err"][:]
        iterations = f.attrs["iterations"]
        true_period = period_set

        if true_period == 0:
            true_period = None

        return data, time, flux, flux_err, true_period, iterations


def make_summary_plots(
    filename, save_fig=False, true_period=None, true_lightcurve=None, lsp=False, models=False, trim_steps=False
):
    """
    Plots and saves all the necessary plots you can get from an hdf5 results file.

    Parameters
    ----------

    filename : hdf5 file
        Name of the file containing the data results.

    true_lightcurve : iterable containing (true_time, true_flux)
        In the case of simulated data, this contains the times and flux values from which the
        simulated data was created (could be higher-resolution than the "data"), useful for
        comparison between models created e.g. from MCMC samples and the true underlying process

    true_period : float
        The true period of the asteroid light curves

    """

    data, time, flux, flux_err, true_period, iterations = read_data(filename, period_set)

    # convert period from log_days to hours
    data[:, :, -1] = np.exp(data[:, :, -1]) * 24.0

    if trim_steps:
        print(trim_steps)
        data = data[:,trim_steps[0]:trim_steps[1],:]

    ###  LOMB-SCARGLE   ###
    ### should be fully functional in both 4 and 6 dim, with period and without

    if lsp:
        print("\nplotting lomb-scargle periodogram")
        run_lsp(
            time, flux, flux_err, data, true_period, true_lightcurve, plot=True
        )

        if save_fig:
            print("saving lomb-scargle periodogram")
            plt.savefig(filename.replace(".hdf5", "_lsp.pdf"), format="pdf")

    ###   TRACE PLOT   ###
    ### should be fully functional in both 4 and 6 dim, with period and without
    # print("\nplotting trace plot")
    # plot_trace(data, iterations)
    #
    # if save_fig:
    #     print("saving trace plot")
    #     plt.savefig(filename.replace(".hdf5", "_trace.pdf"), format="pdf")
    #
    # ###   CORNER PLOTS   ###
    # ### should be fully functional in both 4 and 6 dim, with period and without
    # print("\nplotting corner plot")
    # plot_corner(data, true_period)
    #
    # if save_fig:
    #     print("saving corner plot")
    #     plt.savefig(filename.replace(".hdf5", "_corner.pdf"), format="pdf")
    #
    # print("\nplotting trimmed corner plot")
    # plot_corner(data, true_period, trim=[5, 95])
    #
    # if save_fig:
    #     print("saving trimmed corner plot")
    #     plt.savefig(filename.replace(".hdf5", "_corner_5_95.pdf"), format="pdf")
    #
    # # print("\nplotting zoomed-in corner plot")
    # # plot_corner(data, true_period, zoom=True)
    # #
    # # if save_fig:
    # #     print("saving zoomed-in corner plot")
    # #     plt.savefig(filename.replace(".hdf5", "_corner_zoom.pdf"), format="pdf")

    ###   POSTERIOR   ###
    ### should be fully functional in both 4 and 6 dim, with period and without
    print("\nplotting posterior plot")
    best_period, probs = plot_posterior(data, true_period)

    if save_fig:
        print("saving posterior plot")
        plt.savefig(filename.replace(".hdf5", "_posterior.pdf"), format="pdf")

    print("\nBEST PERIODS")
    print(best_period)

    # ###   FOLDED LIGHTCURVE   ###
    ### should be fully functional in both 4 and 6 dim, with period and without
    print("\nplotting folded lightcurve")
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    if models:
        flat_data = data.reshape(data.shape[0] * data.shape[1], data.shape[2])
        nsamples = flat_data.shape[0]

        nmodels = 3
        npred = 1000

        t_pred = np.linspace(time[0], time[-1], npred)
        m_all = np.zeros((nmodels, t_pred.shape[0]))
        idx = np.random.choice(np.arange(0, nsamples, 1, dtype=int), size=nmodels)

        asteroid = GPFit(time, flux, flux_err, True)
        asteroid.set_params()
        asteroid.set_walker_param_matrix(data.shape[0])
        asteroid.set_gp_kernel()
        gp = asteroid.gp

        for i,j in enumerate(idx):
            p = flat_data[j]
            print(p)
            pnew = [p[0], p[1], p[2], p[3], p[4], np.log(p[5]/24.)] #KEY! need to convert back to log days

            gp.set_parameter_vector(pnew)
            mean_model = gp.sample_conditional(flux, t_pred)
            m_all[i] = mean_model
            models = [t_pred, m_all]
    else:
        models = None

    for i, v in enumerate(best_period):
        plot_folded_lightcurve(
            time,
            flux,
            flux_err=flux_err,
            legend=False,
            ax=ax[int(i/2), i % 2],
            period=best_period[i],
            true_lightcurve=true_lightcurve,
            models = models,
        )

    if save_fig:
        print("saving folded lightcurve")
        plt.savefig(filename.replace(".hdf5", "_folded.pdf"), format="pdf")


def main():

    make_summary_plots(filename, save_fig, period_set, lsp=lsp, models=models, trim_steps=trim_steps)

    return


if __name__ == "__main__":
    ### DEFINE PARSER FOR COMMAND LINE ARGUMENTS
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=" ",  # Bayesian QPO searches for burst light curves.",
        epilog=textwrap.dedent(
            """
    Examples
    --------

    Print this help message:

            $> python run_gp.py --help

    """
        ),
    )

    ### other arguments
    parser.add_argument(
        "-f",
        "--filename",
        action="store",
        dest="filename",
        required=True,
        help="HDF5 file with results.",
    )
    parser.add_argument(
        "-s",
        "--save_fig",
        action="store_true",
        dest="save_fig",
        required=False,
        default=False,
        help="Sets to true if you want to save the figures generated.",
    )
    parser.add_argument(
        "-l",
        "--lsp",
        action="store_true",
        dest="lsp",
        required=False,
        default=False,
        help="Creates an LSP plot.",
    )
    parser.add_argument(
        "-m",
        "--models",
        action="store_true",
        dest="models",
        required=False,
        default=False,
        help="Creates model lightcurves for the folded lightcurve plot.",
    )
    parser.add_argument(
        "-p",
        "--period",
        action="store",
        dest="period",
        required=False,
        type=float,
        help="Set to a value (hours) if you want to plot a known period.",
    )
    parser.add_argument(
        "-t",
        "--trim",
        nargs=2,
        action="store",
        dest="trim",
        required=False,
        type = int,
        help="Set the boundaries for trimming the walkers.",
    )

    clargs = parser.parse_args()

    filename = clargs.filename
    save_fig = clargs.save_fig
    lsp = clargs.lsp
    models = clargs.models
    period_set = clargs.period
    trim_steps = clargs.trim

    main()
