# coding: utf-8

# 1/3 scripts to run

# Import packages

from __future__ import division
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import math
from astropy import table

import params

source_dir = params.source_dir
full_sample = params.full_sample
N_cut = params.N_cut
p_cut = params.p_cut

select = np.load(source_dir + "full_cut.npy")


def load_data(question='t11_arms_number',
              answers=('a31_1', 'a32_2', 'a33_3', 'a34_4',
                       'a36_more_than_4', 'a37_cant_tell')):
    # import the required files (the GZ2 morph, metadata, and Voronoi bins)

    data = table.Table(fits.getdata(source_dir + full_sample, 1))

    morph_cols = ['{}_{}_weighted_fraction'.format(question, a)
                  for a in answers]
    n_morph = len(morph_cols)
    cols = morph_cols + ['PETROMAG_MR', 'R50_KPC', 'REDSHIFT_1']

    # data from the voronoi binning
    bins_names = (['vbin'] +
                  ['zbin_{}'.format(i) for i in range(n_morph)] +
                  ['unknown_{}'.format(i) for i in range(n_morph)] +
                  ['min_fv'] + ['unknown'])
    bins_dtype = ([np.int] * (n_morph + 1) + [np.float] * (n_morph + 2))
    bins = table.Table(np.load(source_dir + 'assignments.npy').T,
                       names=bins_names, dtype=bins_dtype)

    # limit the working dataset to only the columns we need
    # (morphology + binning parameters)
    data = data[cols]

    # cut down the dataset to the selection
    data = data[select]

    # give the vote fraction columns simple, generic names
    for i in range(n_morph):
        data.columns[i].name = 'fv_{}'.format(i)

    # flag galaxies with more than a minimum vote fraction
    # (is this necessary? what is the min_fv?)
    for i in range(n_morph):
        data['flag_{}'.format(i)] = (data['fv_{}'.format(i)] >=
                                     bins.columns['min_fv'])

    # create an index to keep track of the galaxies
    # (is this necessary???)
    idx = table.Table(np.arange(len(data))[:, np.newaxis], names=['index'])

    # produce a combined data table
    data = table.hstack((table.Table(bins.columns[:7]), data, idx))
    data.meta['n_morph'] = n_morph

    # do we really need to return bins as well?
    return data, bins


def plot_raw(ax, D, color):
    # Plot cumulative fractions for the raw data
    ax.plot(D['log10fv'], D['cumfrac'], '-', color=color, lw=2)


def f_logistic(x, k, c):
    # Function to fit the data bin output from the raw plot function
    L = 1 + math.exp(c)
    r = L / (1.0 + np.exp(-k * x + c))
    return r


def f_exp(x, k):
    # Function to fit the data bin output from the raw plot function
    r = np.exp(k * x)
    return r


def f_exp_pow(x, k, c):
    # Function to fit the data bin output from the raw plot function
    r = np.exp(-k * (-x) ** c)
    return r


def chisq_fun(p, f, x, y):
    return ((f(x, *p) - y)**2).sum()


def plot_function(ax, f, x, p, color):
    # Plot fitted function to cumulative fractions
    ax.plot(x, f(x, *p), '--', color=color, lw=0.5)


def plot_guides(ax):
    # Plot guides at 20%, 50%, 80%

    x_guides = np.log10([0.2, 0.5, 0.8])
    y_guides = np.array([0, 1])

    for xg in x_guides:
        ax.plot([xg, xg], y_guides, color=[0, 0, 0], alpha=0.3)


def fit_function(data, bins, plot=True):
    # Output fitted function for each of the Voronoi bins,
    # arm numbers and redshift bins.

    tasklabels = {0: '1', 1: '2', 2: '3', 3: '4', 4: '5+', 5: '??'}

    # Set up the array to write the parameters in to:

    param_data = []

    # Loop over Voronoi magnitude-size bins
    for v in np.unique(bins['vbin']):
        data_v = data[data['vbin'] == v]

        if plot:
            fig, axarr = plt.subplots(2, 3, sharex='col', sharey='row')
            axarr = axarr.ravel()

        # Loop over morphological categories
        for m in range(data.meta['n_morph']):
            z_bins = data_v['zbin_{}'.format(m)]
            z_min = z_bins.min()
            z_max = z_bins.max()
            clr_diff = (1.0 / (z_max - z_min)) if z_max - z_min != 0 else 0

            # Loop over redshift slices
            for z in np.unique(z_bins):
                data_z = data_v[z_bins == z]
                n = len(data_z)
                clr_z = [min((z - 1) * clr_diff, 1), 0,
                         max(1 - (z - 1) * clr_diff, 0)]

                # Compute cumulative fraction
                fv = 'fv_{}'.format(m)
                flag = 'flag_{}'.format(m)
                D = data_z[[fv, flag, 'index']]
                D.sort(fv)
                D['cumfrac'] = np.linspace(0, 1, n)
                D = D[D[flag] == 1]
                # Do we really need to carry around flag?
                # Can we not just do:
                # D = D[D[fv] < 0.00001]
                D['log10fv'] = np.log10(D[fv])

                # Do we need to carry around the index?
                D = D[['log10fv', 'cumfrac', 'index']]

                # Fit function to the cumulative fraction
                # Start fits off in roughly right place with sensible bounds
                # This is still tuned to the arm number question
                if m == 1:
                    func = f_exp_pow
                    p0 = [3, 1]
                    bounds = ((0.5, 10), (0.01, 3))
                else:
                    func = f_logistic
                    p0 = [3, -3]
                    bounds = ((0.5, 6), (-7.5, 0))
                # Note that need to cast x and y to float64 in order
                # for minimisation to work correctly
                res = minimize(chisq_fun, p0,
                               args=(func,
                                     D['log10fv'].astype(np.float64),
                                     D['cumfrac'].astype(np.float64)),
                               bounds=bounds, method='SLSQP')
                p = res.x
                chi2nu = res.fun / (n - len(p))

                if plot:
                    ax = axarr[m]
                    plot_raw(ax, D, clr_z)
                    x = np.linspace(-4, 0, 1000)
                    plot_function(ax, func, x, p, clr_z)

                if len(p) < 2:
                    p = np.array([p[0], 10])

                means = [data_z[c].mean() for c in
                         ['PETROMAG_MR', 'R50_KPC', 'REDSHIFT_1']]
                param_data.append([v, m, z] + means + p[:2].tolist() +
                                  [chi2nu])

            if plot:
                plot_guides(ax)
                ax.tick_params(axis='both', labelsize=10)
                ax.set_xticks(np.arange(5) - 4)
                ax.text(-3.9, 0.9, r'$N_{arms}=$' + tasklabels[m],
                        fontsize=10, ha='left')
                ax.set_ylim([0, 1])
                if m > 2:
                    ax.set_xlabel('r$\log(v_f)$')
                if m in (0, 3):
                    ax.set_ylabel('Cumulative fraction')
                if m == 1:
                    ax.set_title('Voronoi bin %02i' % v)

        if plot:
            fig.savefig('plots/Function_fitting_v{:02d}.pdf'.format(v),
                        dpi=200)
            plt.close()

    # Output parameters for each bin in param_data:

    # 0: v bin
    # 1: a (arm number-1)
    # 2: z bin
    # 3: M_r (mean of bin)
    # 4: R_50 (mean of bin)
    # 5: redshift
    # 6: k (fitted)
    # 7: c (fitted)
    # 8: chi2nu

    param_data = table.Table(np.array(param_data),
                             names=('vbin', 'answer', 'zbin', 'M_r',
                                    'R_50', 'redshift', 'k', 'c', 'chi2nu'))
    return param_data


def save_fit_function(param_data):
    # Save the fitted parameters to a numpy table.
    np.save('npy/fixed_bin_size_params_2.npy', param_data)


if __name__ == '__main__':
    data, bins = load_data()
    param_data = fit_function(data, bins, plot=False)
    save_fit_function(param_data)
