
import numpy as np
from forecast.core import DataGenerator, make_photoz_bin_edges, os, FisherMatrix, time
from lumin_functions.core import def_agn_xlf, def_clusters_xlf
from general_definitions import rep_path
from pathlib import Path
Path("./data").mkdir(parents=True, exist_ok=True)

savepath = rep_path + 'forecast/Fisher_matrices/impact-photoz-cosmology/data/'

fiducial_params = {'Omega_c': 0.25, 'Omega_b': 0.05,
                   'h': 0.7, 'sigma8': 0.8, 'n_s': 0.96,
                   'transfer_function': 'boltzmann_camb',
                   'baryons_power_spectrum': 'nobaryons',
                   'matter_power_spectrum': 'linear'}

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
    'use_weighed_bias': False,
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
                          sigma_0, f_fail,
                          load: bool = True,
                          k_photoz_clu=1.):
    # k_photoz_clu arg needed for analysis of clusters at very low sigma0, e.g 1% or 0.5%
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
        k_photoz = k_photoz_clu
        powspec_pars_dict = powspec_pars_dict_clu
    else:
        raise ValueError('type must be either AGN or Clusters')

    fname = savepath + f'{setname}.npz'

    if load and os.path.isfile(fname):
        arr = np.load(fname, allow_pickle=True)
        pars = arr['pars']
        par_names = arr['par_names']
        F = arr['F']
        J = arr['J']
        #print(f'{setname} DATA LOADED')
        powspec_pars_dict_new = powspec_pars_dict.copy()
        powspec_pars_dict_new['sigma_0'] = sigma_0
        powspec_pars_dict_new['f_fail'] = f_fail
        F = FisherMatrix(par=pars, par_names=par_names, F=F,
                         J=J, name=setname, function=lambda x: x)

        bin_left_edges = make_photoz_bin_edges(
            zmin, zmax, k=k_photoz, sigma_0=powspec_pars_dict_new['sigma_0'])
        powspec_pars_dict_new['bin_left_edges'] = bin_left_edges
        datagen = DataGenerator(RUN_NAME='Fisher_matrices_experiments',
                                fiducial_params=fiducial_params, set_name=setname)
        return F, datagen, powspec_pars_dict_new

    else:

        powspec_pars_dict_new = powspec_pars_dict.copy()
        powspec_pars_dict_new['sigma_0'] = sigma_0
        powspec_pars_dict_new['f_fail'] = f_fail

        bin_left_edges = make_photoz_bin_edges(
            zmin, zmax, k=k_photoz, sigma_0=powspec_pars_dict_new['sigma_0'])
        print(bin_left_edges)
        powspec_pars_dict_new['bin_left_edges'] = bin_left_edges

        datagen = DataGenerator(RUN_NAME='Fisher_matrices_experiments',
                                fiducial_params=fiducial_params, set_name=setname,)

        datagen.invoke(
            **powspec_pars_dict_new, plot_cell=False, plot_dndz=False)
        datagen.invert_cov()
        F = datagen.get_Fisher_matrix()[0]

        np.savez(fname,
                 pars=F.par, par_names=F.par_names,
                 F=F.F, J=F.J,)
        print(f'{datagen.set_name} DATA SAVED')

    return F, datagen, powspec_pars_dict_new


sigmas_all = np.array([0.3, 0.2, 0.1,   0.07, 0.05,
                       0.03, 0.02, 0.015, 0.01, 0.005])
f_fails_all = np.array([0.01, 0.02, 0.05,  0.1, 0.2])
sigmas_agn_ignore = [0.01, 0.005]


if __name__ == '__main__':

    Fs_agn = []
    Fs_clu = []
    idx = 0
    for sigma_0 in sigmas_all:
        if not sigma_0 in sigmas_agn_ignore:
            for f_fail in f_fails_all:
                t0 = time.time()
                print(f'START ===== {sigma_0=} {f_fail=} ====')
                F, _, _ = analyze_sigma0_f_fail(type='AGN', fiducial_params=fiducial_params,
                                                sigma_0=sigma_0, f_fail=f_fail,
                                                load=True)
                Fs_agn.append(F)
                F, _, _ = analyze_sigma0_f_fail(type='Clusters', fiducial_params=fiducial_params,
                                                sigma_0=sigma_0, f_fail=f_fail,
                                                load=True)
                Fs_clu.append(F)

                print(
                    f'DONE ===== {sigma_0=} {f_fail=} in {time.time() - t0} seconds, index = {idx} ====')
                idx += 1

        else:
            for f_fail in f_fails_all:
                if sigma_0 == 0.01:
                    k_photoz_clu = 1
                elif sigma_0 == 0.005:
                    k_photoz_clu = 1.3
                t0 = time.time()
                print(f'START CLUSTERS ONLY ===== {sigma_0=} {f_fail=} ====')
                Fs_agn.append(None)
                F, _, _ = analyze_sigma0_f_fail(type='Clusters', fiducial_params=fiducial_params,
                                                sigma_0=sigma_0, f_fail=f_fail,
                                                load=True, k_photoz_clu=k_photoz_clu)
                Fs_clu.append(F)
                print(
                    f'DONE CLUSTERS ONLY===== {sigma_0=} {f_fail=} in {time.time() - t0} seconds, index = {idx} ====')
                idx += 1
