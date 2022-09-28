import sys
sys.path.append('../../../')


#cosmology
import numpy as np
from scripts.forecast import DataGenerator, make_photoz_bin_edges, os, time, FisherMatrix
from scripts.luminosity_functions import def_agn_xlf, def_clusters_xlf
from scripts.utils import path2res_forecast


savepath = path2res_forecast + 'cosmo-photoz/'


#cosmology
fiducial_params = {'Omega_c': 0.25, 'Omega_b': 0.05,
                   'h': 0.7, 'sigma8': 0.8, 'n_s': 0.96,
                   'transfer_function': 'boltzmann_camb',
                   'baryons_power_spectrum': 'nobaryons',
                   'matter_power_spectrum': 'linear'}

#survey parameters
fsky = 0.658
powspec_pars_dict_agn = {
    'slim': 1e-14,
    'l_min': 10,
    'l_max': 500,
    'log_bins': 41,
    'fsky': fsky,
    'has_rsd': True,
    'density_multiplier': 1.3,
    'camb_llimber': 110,
    'xlf': def_agn_xlf,
    'use_camb': True,
    'delta_i': 3,
    'remove_ignored_cells': True,
}


powspec_pars_dict_clu = {
    'slim': 4.4e-14,
    'l_min': 10,
    'l_max': 150,
    'log_bins': 41,
    'fsky':  fsky,
    'has_rsd': True,
    'density_multiplier': 1,
    'camb_llimber': 110,
    'xlf': def_clusters_xlf,
    'use_camb': True,
    'delta_i': 3,
    'remove_ignored_cells': True,
}
zmin_clu = 0.1
zmax_clu = 0.8
k_photoz_clu = 1

zmin_agn = 0.5
zmax_agn = 2.5
k_photoz_agn = 1


def analyze_sigma0_f_fail(type, fiducial_params,
                          sigma_0, f_fail,):


    if type == 'AGN':
        setname = f'AGN_{sigma_0}_{f_fail}'
        zmin = zmin_agn
        zmax = zmax_agn
        k_photoz = k_photoz_agn
        powspec_pars_dict = powspec_pars_dict_agn
    elif type == 'Clusters':
        setname = f'Clusters_{sigma_0}_{f_fail}'
        zmin = zmin_clu
        zmax = zmax_clu
        if sigma_0 == 0.005:
            k_photoz = k_photoz_clu*1.3
        else:
            k_photoz = k_photoz_clu
        powspec_pars_dict = powspec_pars_dict_clu
    else:
        raise ValueError('type must be either AGN or Clusters')

    
    powspec_pars_dict_new = powspec_pars_dict.copy()
    powspec_pars_dict_new['sigma_0'] = sigma_0
    powspec_pars_dict_new['f_fail'] = f_fail

    bin_left_edges = make_photoz_bin_edges(
        zmin, zmax, k=k_photoz, sigma_0=powspec_pars_dict_new['sigma_0'])
    print(bin_left_edges)
    powspec_pars_dict_new['bin_left_edges'] = bin_left_edges

    datagen = DataGenerator(fiducial_params=fiducial_params, set_name=setname,)

    datagen.invoke(
        **powspec_pars_dict_new, plot_cell=False, plot_dndz=False)
    datagen.invert_cov()
    F = datagen.get_Fisher_matrix()[0]

    return F, datagen, powspec_pars_dict_new


sigmas_clu = np.array([0.3, 0.2, 0.1,   0.07, 0.05,
                       0.03, 0.02, 0.015, 0.01, 0.005])
f_fails_clu = np.array([0.01, 0.02, 0.05,  0.1, 0.2])


sigmas_agn = np.array([0.3, 0.2, 0.1,   0.07, 0.05,
                       0.03, 0.02, 0.015])
f_fails_agn = np.array([0.01, 0.02, 0.05,  0.1, 0.2])



if __name__ == '__main__':

    #AGN
    for sigma_0 in sigmas_agn:
        for f_fail in f_fails_agn:
            fname_cosmo_agn = savepath + \
                f'AGN_{sigma_0}_{f_fail}.npz'
            if os.path.isfile(fname_cosmo_agn):
                print(f' {fname_cosmo_agn} exists')
            else:
                print(f'Analyzing AGN sigma_0={sigma_0}, f_fail={f_fail}')

                F, _,_ = analyze_sigma0_f_fail('AGN', fiducial_params, sigma_0, f_fail)
                np.savez(fname_cosmo_agn,
                            pars=F.par, par_names=F.par_names,
                            F=F.F, J=F.J,)
                print(f'{fname_cosmo_agn} DATA SAVED')

    #clusters

    for sigma_0 in sigmas_clu:
        for f_fail in f_fails_clu:
            fname_cosmo_clu = savepath + \
                f'Clusters_{sigma_0}_{f_fail}.npz'
            if os.path.isfile(fname_cosmo_clu):
                print(f' {fname_cosmo_clu} exists')
            else:
                print(f'Analyzing Clusters sigma_0={sigma_0}, f_fail={f_fail}')

                F, _,_ = analyze_sigma0_f_fail('Clusters', fiducial_params, sigma_0, f_fail)
                np.savez(fname_cosmo_clu,
                            pars=F.par, par_names=F.par_names,
                            F=F.F, J=F.J,)
                print(f'{fname_cosmo_clu} DATA SAVED')

