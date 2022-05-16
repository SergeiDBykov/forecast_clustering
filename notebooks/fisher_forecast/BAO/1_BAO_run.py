import sys
sys.path.append('../../../')

#importrs 
import numpy as np
from scripts.forecast import DataGenerator, make_photoz_bin_edges, os, time
from scripts.luminosity_functions import def_agn_xlf, def_clusters_xlf
from scripts.utils import path2res_forecast

savepath = path2res_forecast + 'BAO/'


#cosmology
fiducial_params_wiggle = {'Omega_c': 0.25, 'Omega_b': 0.05,
                          'h': 0.7, 'sigma8': 0.8, 'n_s': 0.96,
                          'transfer_function': 'eisenstein_hu',
                          'baryons_power_spectrum': 'nobaryons',
                          'matter_power_spectrum': 'linear'}

fiducial_params_no_wiggle = {'Omega_c': 0.25, 'Omega_b': 0.05,
                             'h': 0.7, 'sigma8': 0.8, 'n_s': 0.96,
                             'transfer_function': 'eisenstein_hu_nowiggles',
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
    'use_weighed_bias': False,
    'density_multiplier': 1.3,
    'camb_llimber': 110,
    'xlf': def_agn_xlf,
    'use_camb': False,
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
    'use_weighed_bias': False,
    'density_multiplier': 1,
    'camb_llimber': 110,
    'xlf': def_clusters_xlf,
    'use_camb': False,
    'delta_i': 3,
    'remove_ignored_cells': True,
}
zmin_clu = 0.1
zmax_clu = 0.8
k_photoz_clu = 1

zmin_agn = 0.5
zmax_agn = 2.5
k_photoz_agn = 1


def analyze_sigma0_f_fail(type, fiducial_params: dict,
                          sigma_0: float, f_fail: float,
                          k_photoz_clu: float =1.):
    """
    analyze_sigma0_f_fail for a given fiducial_params, sigma_0 and f_fail returns a DataGenerator object from which you can access Cell and covariance matrix objects.

    Args:
        type (str): Type of a tracer: AGN of Clusters
        fiducial_params (dict): dictionary of invocation parameters for the DataGenerator
        sigma_0 (float): photo-z scatter
        f_fail (float): fraction of catastrophic failures of photozs
        k_photoz_clu (float, optional): k_photo parameter. Defaults to 1..

    """
    # k_photoz_clu arg needed for analysis of clusters at very low sigma0, e.g 1% or 0.5%
    transfer_name = fiducial_params['transfer_function']
    if type == 'AGN':
        setname = f'AGN_{sigma_0}_{f_fail}_{transfer_name}'
        zmin = zmin_agn
        zmax = zmax_agn
        k_photoz = k_photoz_agn
        powspec_pars_dict = powspec_pars_dict_agn
    elif type == 'Clusters':
        setname = f'Clusters_{sigma_0}_{f_fail}_{transfer_name}'
        zmin = zmin_clu
        zmax = zmax_clu
        k_photoz = k_photoz_clu
        powspec_pars_dict = powspec_pars_dict_clu
    else:
        raise ValueError('type must be either AGN or Clusters')


    powspec_pars_dict_new = powspec_pars_dict.copy()
    powspec_pars_dict_new['sigma_0'] = sigma_0
    powspec_pars_dict_new['f_fail'] = f_fail

    bin_left_edges = make_photoz_bin_edges(
        zmin, zmax, k=k_photoz, sigma_0=powspec_pars_dict_new['sigma_0'])
    powspec_pars_dict_new['bin_left_edges'] = bin_left_edges

    datagen = DataGenerator(RUN_NAME='Fisher_matrices_experiments',
                            fiducial_params=fiducial_params, set_name=setname,)

    datagen.invoke(
        **powspec_pars_dict_new, plot_cell=False, plot_dndz=False)


    return datagen.cls_rebin_lkl, datagen.cov_rebin_lkl


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
            fname_bao_agn = savepath + \
                f'AGN_{sigma_0}_{f_fail}_bao_sign.npz'
            if os.path.isfile(fname_bao_agn):
                print(f' {fname_bao_agn} exists')
            else:
                print(f'Analyzing AGN sigma_0={sigma_0}, f_fail={f_fail}')

                data_agn, cov_agn = analyze_sigma0_f_fail(type='AGN', fiducial_params=fiducial_params_wiggle,
                                                            sigma_0=sigma_0, f_fail=f_fail)

                data_agn_no_bao, cov_agn_no_bao = analyze_sigma0_f_fail(type='AGN', fiducial_params=fiducial_params_no_wiggle,
                                                                        sigma_0=sigma_0, f_fail=f_fail,)

                diff = data_agn - data_agn_no_bao
                icov = np.linalg.inv(cov_agn)  # type: ignore
                bao_sign_agn = np.sqrt(np.einsum('i, ij, j ->', diff, icov, diff))
                np.savez(fname_bao_agn,
                        bao_sign=bao_sign_agn)


    #Clusters
    for sigma_0 in sigmas_clu:
        for f_fail in f_fails_clu:

            fname_bao_clu = savepath + \
                f'Clusters_{sigma_0}_{f_fail}_bao_sign.npz'
           
            if os.path.isfile(fname_bao_clu):
                print(f' {fname_bao_clu} exists')
             
            else:
                print(f'Analyzing Clusters sigma_0={sigma_0}, f_fail={f_fail}')

                data_clu, cov_clu = analyze_sigma0_f_fail(type='Clusters', fiducial_params=fiducial_params_wiggle,
                                                            sigma_0=sigma_0, f_fail=f_fail)

                data_clu_no_bao, cov_clu_no_bao = analyze_sigma0_f_fail(type='Clusters', fiducial_params=fiducial_params_no_wiggle,
                                                                        sigma_0=sigma_0, f_fail=f_fail,)

                diff = data_clu - data_clu_no_bao
                icov = np.linalg.inv(cov_clu)  # type: ignore
                bao_sign_clu = np.sqrt(np.einsum('i, ij, j ->', diff, icov, diff))
                np.savez(fname_bao_clu,
                        bao_sign=bao_sign_clu)
