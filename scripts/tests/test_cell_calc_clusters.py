import sys
sys.path.append('../../')


from functools import lru_cache
import warnings
import matplotlib
from scripts.utils import np, plt
from scripts.luminosity_functions import cgs_flux, def_clusters_xlf
from scripts.forecast import dNdz_photo_z, DensityTracers, DataGenerator, cl_get_ij_from_idx, _set_ccl_cosmo_to_camb_cosmo_, transform_ccl_tracers_to_camb, logrebin_aps, Cell_calculator, cov_Cell, noise_Cell, sparse_arrays, gaussian_loglike, camb, ccl
import jax_cosmo as jc

fiducial_params = {'Omega_c': 0.25, 'Omega_b': 0.05,
                   'h': 0.7, 'sigma8': 0.8, 'n_s': 0.96,
                   # 'transfer_function': 'eisenstein_hu',
                   'transfer_function': 'boltzmann_camb',
                   'baryons_power_spectrum': 'nobaryons',
                   'matter_power_spectrum': 'linear'}

powspec_pars_dict = {
    'bin_left_edges': np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
    'sigma_0': 0.01,
    'f_fail': 0.02,
    'slim': 4.4e-14,
    'l_min': 5,
    'l_max': 150,
    'log_bins': -1,
    'fsky': 0.7,
    'has_rsd': True,
    'delta_i': -1,
    'use_camb': True,
    'camb_llimber': 110,
    'remove_ignored_cells': False,
    'density_multiplier': 1,
}


def make_datagen(fiducial_params, powspec_pars_dict):

    datagen = DataGenerator(
        fiducial_params=fiducial_params,
        set_name='TEST PROCEDURES')

    datagen.gen_dNdz(bin_left_edges=powspec_pars_dict['bin_left_edges'],
                     f_fail=powspec_pars_dict['f_fail'],
                     sigma_0=powspec_pars_dict['sigma_0'],
                     xlf=def_clusters_xlf,
                     slim=powspec_pars_dict['slim'],
                     density_multiplier=powspec_pars_dict['density_multiplier'],)

    datagen.make_tracers(has_rsd=powspec_pars_dict['has_rsd'])

    datagen.gen_Cell(
        l_min=powspec_pars_dict['l_min'],
        l_max=powspec_pars_dict['l_max'],
        log_bins=powspec_pars_dict['log_bins'],
        fsky=powspec_pars_dict['fsky'],
        delta_i=powspec_pars_dict['delta_i'],
        use_camb=powspec_pars_dict['use_camb'],
        camb_llimber=powspec_pars_dict['camb_llimber'],
        remove_ignored_cells=powspec_pars_dict['remove_ignored_cells'],
    )
    return datagen


# @lru_cache(maxsize=None)
def gen_dndz(args):
    xlf, slim, zarr = args
    return xlf.dNdz(Slim=slim, zarr=zarr)[1]


zarr = np.linspace(0.05, 1.5, 750)
dNdz_precomputed = gen_dndz(
    (def_clusters_xlf, powspec_pars_dict['slim'], zarr))


def cell_suite(fiducial_params, powspec_pars_dict):
    datagen = make_datagen(fiducial_params, powspec_pars_dict)
    # unpack powspec_pars_dict inti variables:
    bin_left_edges = powspec_pars_dict['bin_left_edges']
    sigma_0 = powspec_pars_dict['sigma_0']
    f_fail = powspec_pars_dict['f_fail']
    slim = powspec_pars_dict['slim']
    density_multiplier = powspec_pars_dict['density_multiplier']
    use_camb = powspec_pars_dict['use_camb']
    cosmology = ccl.Cosmology(**fiducial_params)  # type: ignore

    # dNdz = def_agn_xlf.dNdz(Slim=slim, zarr=zarr)[1]
    #dNdz=gen_dndz((def_agn_xlf, slim, zarr))
    dNdz = dNdz_precomputed.copy()
    dndz_photoz = [dNdz_photo_z(zarr=zarr, dNdz=dNdz, zmin=bin_left_edges[i], zmax=bin_left_edges[i+1],
                                sigma_0=sigma_0, f_fail=f_fail)[1] for i in range(len(bin_left_edges)-1)]
    n_obj_bin = [dNdz_photo_z(zarr=zarr, dNdz=dNdz, zmin=bin_left_edges[i], zmax=bin_left_edges[i+1],
                              sigma_0=sigma_0, f_fail=f_fail)[2] for i in range(len(bin_left_edges)-1)]
    tracers_obj = DensityTracers(
        zarr_list=datagen.zarrs, dndz_list=dndz_photoz, bias_list=list(datagen.bias_arrs))

    trs, _ = tracers_obj.make_tracers(
        cosmo_ccl=cosmology, has_rsd=powspec_pars_dict['has_rsd'])

    for i in range(len(bin_left_edges)-1):
        assert np.allclose(datagen.dndz_arrs[i], dndz_photoz[i] *
                           density_multiplier, atol=0, rtol=1e-4), f'dndz differ! {i=}'

    assert np.allclose(datagen.bias_arr, def_clusters_xlf.b_eff(
        zarr=zarr, Slim=powspec_pars_dict['slim'])[1], atol=0, rtol=1e-4), f'biases b_eff(z) differ!'
    for i in range(len(bin_left_edges)-1):
        assert np.allclose(np.array(datagen.src_dens_list), np.array(
            n_obj_bin)*density_multiplier, atol=0, rtol=1e-4), f'src dens differs! {i=}'

    for i in range(len(bin_left_edges)-1):
        assert np.allclose(trs[i].dndz_arr, datagen.dndz_arrs[i] /
                           density_multiplier, atol=0, rtol=1e-4), f'dndz differ! {i=}'
        assert np.allclose(trs[i].dndz_arr, dndz_photoz[i],
                           atol=0, rtol=1e-4), f'dndz differs! {i=}'

    assert np.allclose(np.array(datagen.src_dens_list), np.array(tracers_obj.src_dens_list) *
                       density_multiplier), f'src dens differs for DensityTracers object: {datagen.src_dens_list=}, {datagen.src_dens_list=} '

    for i in range(len(bin_left_edges)-1):
        assert np.allclose(trs[i].z_arr, datagen.tracers[i].z_arr,
                           atol=0, rtol=1e-4), f'zarrs differ! {i=}'
        assert np.allclose(trs[i].dndz_arr*density_multiplier,
                           datagen.tracers[i].dndz_arr, atol=0, rtol=1e-4), f'dndz_arr differ! {i=}'
        assert np.allclose(trs[i].bias_arr, datagen.tracers[i].bias_arr,
                           atol=0, rtol=1e-4), f'bias_arr differ! {i=}'
        assert trs[i].has_rsd == datagen.tracers[
            i].has_rsd, f'has_rsd differ! {i=}'
        print('Tracers test passed')

        if use_camb:
            for idx in range(datagen.cls_rebin.shape[0]):
                i, j = cl_get_ij_from_idx(idx, datagen.n_bins)

                cp = camb.model.CAMBparams()
                cp = _set_ccl_cosmo_to_camb_cosmo_(cosmology, cp)
                camb_tracers = transform_ccl_tracers_to_camb(trs)
                cp.SourceWindows = camb_tracers
                cp.set_for_lmax(powspec_pars_dict['l_max'])
                cp.SourceTerms.limber_windows = True
                cp.SourceTerms.limber_phi_lmin = datagen.camb_llimber
                cp.SourceTerms.counts_redshift = trs[0].has_rsd  # type: ignore
                results = camb.get_results(cp)
                cls_res = results.get_source_cls_dict(raw_cl=True)
                ell = np.arange(2, powspec_pars_dict['l_max']+1)
                sigma8_camb = results.get_sigma8_0()
                norm_s8 = (cosmology['sigma8'] / sigma8_camb)**2
                camb_idx = f"W{i+1}xW{j+1}"
                ccl_tmp = cls_res[camb_idx][2:powspec_pars_dict['l_max']+1]*norm_s8
                ccl_tmp = ccl_tmp[ell >= powspec_pars_dict['l_min']]
                cij = ccl_tmp

                cij = logrebin_aps(
                    ell=ell, cls=cij, log_bins=powspec_pars_dict['log_bins'])[1]

                assert np.allclose(
                    datagen.cls_rebin[idx], cij, atol=0, rtol=1e-4), f'cls differ! {i=} {j=}'
            print('CAMB Cell test passed')
        else:
            for idx in range(datagen.cls_rebin.shape[0]):
                i, j = cl_get_ij_from_idx(idx, datagen.n_bins)
                ell_ccl = np.arange(
                    powspec_pars_dict['l_min'], powspec_pars_dict['l_max']+1)
                cij = ccl.angular_cl(  # type: ignore
                    cosmology, trs[i], trs[j], ell=ell_ccl)
                cij = logrebin_aps(
                    ell=ell_ccl, cls=cij, log_bins=powspec_pars_dict['log_bins'])[1]

                assert np.allclose(
                    datagen.cls_rebin[idx], cij, atol=0, rtol=1e-4), f'cls differ! {i=} {j=}'
            print('CCL Cell test passed')


def data_removal_suite(delta_i=2, error_mag=0.1):
    pars_no_ignore = powspec_pars_dict.copy()
    pars_no_ignore['delta_i'] = delta_i
    pars_no_ignore['remove_ignored_cells'] = False
    pars_no_ignore['log_bins'] = 26

    pars_ignore = powspec_pars_dict.copy()
    pars_ignore['delta_i'] = delta_i
    pars_ignore['remove_ignored_cells'] = True
    pars_ignore['log_bins'] = 26

    dg_full = make_datagen(fiducial_params=fiducial_params,
                           powspec_pars_dict=pars_no_ignore)
    dg_full.invert_cov()
    dg_ignore = make_datagen(
        fiducial_params=fiducial_params, powspec_pars_dict=pars_ignore)
    dg_ignore.invert_cov()
    assert np.allclose(dg_full.snr, dg_ignore.snr,
                       atol=0, rtol=1e-4), 'snr differ!'

    data_full = dg_full.cls_rebin_lkl
    idx = data_full != 0
    error_mag = error_mag
    model_full = np.zeros_like(data_full)
    model_full[idx] = np.random.normal(
        loc=data_full[idx], scale=np.abs(error_mag*data_full[idx]), size=data_full[idx].shape)
    diff_full = data_full - model_full
    icov_full = dg_full.inv_cov_rebin_lkl
    chi2_full = np.sqrt(
        np.einsum('i, ji,j->', diff_full, icov_full, diff_full))

    model_ignore = model_full[idx]
    data_ignore = dg_ignore.cls_rebin_lkl
    diff_ignore = data_ignore - model_ignore
    icov_ignore = dg_ignore.inv_cov_rebin_lkl
    chi2_ignore = np.sqrt(
        np.einsum('i, ji,j->', diff_ignore, icov_ignore, diff_ignore))

    assert np.allclose(chi2_full, chi2_ignore, atol=0,
                       rtol=1e-2), f'chi2_ignore is not close to chi2: {chi2_full}, {chi2_ignore}'
    print('Data removal test passed')


def jacobian_suite(fiducial_params, powspec_pars_dict, step=5e-4):
    datagen = make_datagen(fiducial_params=fiducial_params,
                           powspec_pars_dict=powspec_pars_dict)
    datagen.invert_cov()
    Fs = datagen.get_Fisher_matrix(jac_step=step)
    F = Fs[0]

    def calc_deriv(datagen, par_idx, step):
        par_fiducial = datagen.par_vector
        def func(x): return datagen.Cell_mean(x)[1]
        par_new = par_fiducial.copy()
        par_new[par_idx] += step
        return (func(par_new) - func(par_fiducial))/step

    my_J = []
    for i in range(len(datagen.par_vector)):
        deriv = calc_deriv(datagen, i, step=step)
        my_J.append(deriv)
    my_J = np.array(my_J).T
    MyF = np.dot(my_J.T, np.dot(datagen.inv_cov_rebin_lkl, my_J))

    for i in range(len(datagen.par_vector)):
        deriv = my_J[:, i]
        deriv_jac = F.J[:, i]

        if not np.allclose(
                deriv, deriv_jac, atol=2e-7, rtol=0.05):

            warnings.warn(
                f'derivatives do not match, {deriv/deriv_jac=}, {i=}, {deriv[np.argmax(deriv/deriv_jac)]}')

    assert np.allclose(
        F.F, MyF, atol=0, rtol=5e-2), f'Fisher matrices do not match, {F.F/MyF=}'


# test 1: standard
print('######## STANDARD TESTS #########')
cell_suite(fiducial_params=fiducial_params,
           powspec_pars_dict=powspec_pars_dict,)
ccl_calc = powspec_pars_dict.copy()
ccl_calc['use_camb'] = False
cell_suite(fiducial_params=fiducial_params,
           powspec_pars_dict=ccl_calc,)



# test 2: change cosmo
print('######## CHANGE COSMO TESTS #########')
cosmo1 = fiducial_params.copy()
cosmo1['h'] = 0.6
cosmo2 = fiducial_params.copy()
cosmo2['Omega_c'] = 0.3
cell_suite(fiducial_params=cosmo1,
           powspec_pars_dict=powspec_pars_dict)
cell_suite(fiducial_params=cosmo2,
           powspec_pars_dict=powspec_pars_dict)


#  test 3: change dndz: bin_left_edges and photoz
print('######## CHANGE DNDZ TESTS #########')
bins_1 = powspec_pars_dict.copy()
bins_1['bin_left_edges'] = np.array([0.1, 0.15, 0.25, 0.3])
bins_1['sigma_0'] = 0.04

bins_2 = powspec_pars_dict.copy()
bins_2['bin_left_edges'] = np.array([0.2, 0.5, 0.7])
bins_2['f_fail'] = 0.3

cell_suite(fiducial_params=fiducial_params,
           powspec_pars_dict=bins_1)

cell_suite(fiducial_params=fiducial_params,
           powspec_pars_dict=bins_2)


# test 4: l binning
print('######## L BINNING TESTS #########')
binning_1 = powspec_pars_dict.copy()
binning_1['l_min'] = 5
binning_1['l_max'] = 250
binning_1['log_bins'] = -1

binning_2 = powspec_pars_dict.copy()
binning_2['log_bins'] = 16
binning_2['l_max'] = 400

cell_suite(fiducial_params=fiducial_params,
           powspec_pars_dict=binning_1)


# test 5: rsd
print('######## RSD TESTS #########')
rsd_1 = powspec_pars_dict.copy()
rsd_1['has_rsd'] = True

cell_suite(fiducial_params=fiducial_params,
           powspec_pars_dict=rsd_1)


# test 6: ignoring stuff
print('######## DATA IGNORING TESTS #########')

data_removal_suite(delta_i=2, error_mag=0.1)
data_removal_suite(delta_i=3, error_mag=0.1)
data_removal_suite(delta_i=2, error_mag=0.01)
data_removal_suite(delta_i=2, error_mag=0.5)
data_removal_suite(delta_i=0, error_mag=0.5)
data_removal_suite(delta_i=10, error_mag=0.5)


# test 7: jacobian
print('######## JACOBIAN TESTS #########')
jacobian_suite(fiducial_params=fiducial_params,
               powspec_pars_dict=powspec_pars_dict)
jacobian_suite(fiducial_params=cosmo1,
               powspec_pars_dict=powspec_pars_dict)

jacobian_suite(fiducial_params=cosmo2,
               powspec_pars_dict=binning_2, step=5e-4)
# TODO step of jacobian depends on the cosmology. in the example above with O_m = 0.3, step = 2e-3 produces bad derivatives
# jacobian_suite(fiducial_params=fiducial_params,
#               powspec_pars_dict=full_bias, step=1e-3)


print('######## ALL TESTS PASSED #########')
