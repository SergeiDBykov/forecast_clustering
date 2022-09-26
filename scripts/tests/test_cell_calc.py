import sys
sys.path.append('../../')

from functools import lru_cache
import warnings
import matplotlib
from scripts.utils import np, plt
from scripts.luminosity_functions import def_agn_xlf, cgs_flux
from scripts.forecast import dNdz_photo_z, DensityTracers, DataGenerator, cl_get_ij_from_idx, _set_ccl_cosmo_to_camb_cosmo_, transform_ccl_tracers_to_camb, logrebin_aps, Cell_calculator, cov_Cell, noise_Cell, sparse_arrays, gaussian_loglike, camb, ccl
import jax_cosmo as jc


fiducial_params = {'Omega_c': 0.25, 'Omega_b': 0.05,
                   'h': 0.7, 'sigma8': 0.8, 'n_s': 0.96,
                   # 'transfer_function': 'eisenstein_hu',
                   'transfer_function': 'boltzmann_camb',
                   'baryons_power_spectrum': 'nobaryons',
                   'matter_power_spectrum': 'linear'}

powspec_pars_dict = {
    'bin_left_edges': [0.5, 0.6, 0.7, 1,  1.2],
    'sigma_0': 0.05,
    'f_fail': 0.1,
    'slim': 1.2e-14,
    'l_min': 10,
    'l_max': 520,
    'log_bins': -1,
    'fsky': 0.7,
    'has_rsd': False,
    'delta_i': -1,
    'use_camb': True,
    'camb_llimber': 110,
    'remove_ignored_cells': False,
    'density_multiplier': 1.3,
}


def cell_calculator_function_suite():

    cosmo_ccl_s8 = ccl.Cosmology(
        Omega_c=0.3, Omega_b=0.05, h=0.7, sigma8=0.8, n_s=0.96, Neff=0,
        transfer_function='eisenstein_hu', matter_power_spectrum='linear')
    cosmo_jax = jc.Cosmology(Omega_c=0.3, Omega_b=0.05, h=0.7,
                             sigma8=0.8, n_s=0.96, Omega_k=0., w0=-1., wa=0.)

    # %% create tracers
    z = np.linspace(0, 4, 100)

    nz1 = jc.redshift.smail_nz(1., 2, 0.3, gals_per_arcmin2=1000/3600)
    nz2 = jc.redshift.smail_nz(1., 2, 0.6, gals_per_arcmin2=2000/3600)
    nz3 = jc.redshift.smail_nz(1., 2, 0.8, gals_per_arcmin2=500/3600)

    zarrs = [z, z, z]
    dndz_arrs = [nz1(z)*1000, nz2(z)*2000, nz3(z)*500]
    tracers = DensityTracers(zarrs, dndz_arrs, bias_list=[1, 1, 1])
    tracers.plot_dNdz(lw=3, alpha=0.7,)

    # %% calculate cel l

    trs, cosmo = tracers.make_tracers(cosmo_ccl_s8)
    noise_power = noise_Cell(tracers.src_dens_list)  # tracers.noise_Cell()

    nsteps = 0
    fsky = 0.9
    ell_rebin, cls_rebin, n_logbin, _ = Cell_calculator(
        l_min=10, l_max=300, cosmo_ccl=cosmo_ccl_s8, tracers=trs, log_bins=nsteps, delta_i=-1, use_camb=False)
    cov_rebin = cov_Cell(cls_rebin, noise_power, ell_rebin, fsky, n_logbin)
    cls_rebin_lkl, cov_rebin_lkl = sparse_arrays(cls_rebin, cov_rebin)

    # check covariance matrix
    cls_rebin_noisy = (cls_rebin.T + noise_power).T
    cov_norm = (2*ell_rebin+1)*fsky

    for j in [0, 3, 5]:
        assert np.allclose(((cls_rebin_noisy[j]*cls_rebin_noisy[j]+cls_rebin_noisy[j] * cls_rebin_noisy[j])
                            )/cov_norm,  cov_rebin[j][j], atol=0, rtol=1e-5), f'cov check for autocorr failed at {j=}'

    assert np.allclose(((cls_rebin_noisy[1]*cls_rebin_noisy[0]+cls_rebin_noisy[0]
                         * cls_rebin_noisy[1]))/cov_norm,  cov_rebin[0][1], atol=0, rtol=1e-5)

    assert np.allclose(((cls_rebin_noisy[2]*cls_rebin_noisy[0]+cls_rebin_noisy[0]
                         * cls_rebin_noisy[2]))/cov_norm,  cov_rebin[0][2], atol=0, rtol=1e-5)

    assert np.allclose(((cls_rebin_noisy[1]*cls_rebin_noisy[1]+cls_rebin_noisy[1]
                         * cls_rebin_noisy[1]))/cov_norm,  cov_rebin[0][3], atol=0, rtol=1e-5)

    assert np.allclose(((cls_rebin_noisy[1]*cls_rebin_noisy[2]+cls_rebin_noisy[1]
                         * cls_rebin_noisy[2]))/cov_norm,  cov_rebin[0][4], atol=0, rtol=1e-5)

    assert np.allclose(((cls_rebin_noisy[2]*cls_rebin_noisy[2]+cls_rebin_noisy[2]
                         * cls_rebin_noisy[2]))/cov_norm,  cov_rebin[0][5], atol=0, rtol=1e-5)

    assert np.allclose(((cls_rebin_noisy[0]*cls_rebin_noisy[3]+cls_rebin_noisy[1]
                         * cls_rebin_noisy[1]))/cov_norm,  cov_rebin[1][1], atol=0, rtol=1e-5)

    assert np.allclose(((cls_rebin_noisy[0]*cls_rebin_noisy[4]+cls_rebin_noisy[1]
                         * cls_rebin_noisy[2]))/cov_norm,  cov_rebin[1][2], atol=0, rtol=1e-5)

    assert np.allclose(((cls_rebin_noisy[1]*cls_rebin_noisy[3]+cls_rebin_noisy[1]
                         * cls_rebin_noisy[3]))/cov_norm,  cov_rebin[1][3], atol=0, rtol=1e-5)

    assert np.allclose(((cls_rebin_noisy[1]*cls_rebin_noisy[3]+cls_rebin_noisy[1]
                         * cls_rebin_noisy[3]))/cov_norm,  cov_rebin[1][3], atol=0, rtol=1e-5)

    # %% compare with CCL

    for jj in range(6):
        i, j = cl_get_ij_from_idx(jj, 3)
        # # fig,  [ax, ax_rat] = plt.subplots(2, figsize=(4, 8))
        # fig,  ax = plt.subplots(figsize=(8, 8))
        # make_error_boxes(ax, ell_rebin, cls_rebin[jj], n_logbin/2,
        #                  cov_rebin[jj, jj]**0.5, lw=4, label=f'my code: {i}x{j}', alpha=0.4)
        # ax.set_xlabel(r'$\ell$')
        # ax.set_ylabel(r'$C_\ell$')

        ell = np.arange(10, 300+1)
        clu1, clu2 = trs[i], trs[j]
        cls_clu = ccl.angular_cl(cosmo, clu1, clu2, ell)
        ell_rebin, cls_clu_reb, _, _ = logrebin_aps(
            ell, cls_clu,  log_bins=nsteps)
        # ax.plot(ell, cls_clu, 'k-.', lw=1, label=f'raw ccl: {i}x{j}')
        # ax.legend()

        # ax_rat.plot(ell_rebin, cls_clu_reb /
        #            cls_rebin[jj], lw=4, label=f'raw ccl/ my code,  {i}x{j}')

        # fig.savefig(f'plots/ccl_comparison_{i}x{j}.png')
        assert np.allclose(cls_rebin[jj], cls_clu_reb, atol=0,
                           rtol=1e-4), f'codes differ. {cls_rebin[jj]} != {cls_clu_reb}'

    # %% compare with jax cosmo

    nzs = [nz1, nz2, nz3]
    ell = np.arange(10, 300+1)
    probes = [jc.probes.NumberCounts(nzs, jc.bias.constant_linear_bias(1.))]
    mu_jax, cov_jax = jc.angular_cl.gaussian_cl_covariance_and_mean(
        cosmo_jax, ell, probes, sparse=True, f_sky=0.9, nonlinear_fn=jc.power.linear)
    cls_jax = mu_jax.reshape((6, -1))
    cov_jax_2d = jc.sparse.to_dense(cov_jax)

    for jj in range(6):
        # fig,  ax = plt.subplots(figsize=(8, 8))
        i, j = cl_get_ij_from_idx(jj, 3)
        jax_test_rebin = logrebin_aps(ell, cls_jax[jj], log_bins=nsteps)[1]
        # ax.loglog(ell_rebin, jax_test_rebin,
        #          label=f'jax {i}x{j}', lw=3, color='k')
        # if nsteps <= 0:
        #    ax.errorbar(ell_rebin, jax_test_rebin,
        #                cov_jax[jj, jj]**0.5, fmt='none', ecolor='k', alpha=0.4)
        # ax.loglog(ell_rebin, cls_rebin[jj], ls='-.',
        #          label=f'ccl {i}x{j}', color='g')
        # ax.errorbar(ell_rebin+1, cls_rebin[jj], cov_rebin[jj, jj]
        #            ** 0.5, fmt='none', ecolor='g', alpha=0.4)

        # plt.legend()

    if nsteps <= 0:
        assert(np.allclose(cls_rebin, cls_jax, atol=0, rtol=1e-2))
        assert(np.allclose(cov_rebin, cov_jax, atol=0, rtol=1e-2))
        assert(np.allclose(cov_rebin_lkl, cov_jax_2d, atol=0, rtol=1e-2))
    tiny = np.min(cov_rebin_lkl[cov_rebin_lkl > 0])
    # plt.figure()
    # rat = (cov_rebin_lkl+tiny) / (cov_jax_2d+tiny)
    # plt.imshow(rat)
    # plt.colorbar()
    if cls_rebin_lkl.shape == mu_jax.shape:
        model = np.random.normal(mu_jax, 0.1*mu_jax)  # type: ignore

        lkl_jax = jc.likelihood.gaussian_log_likelihood(
            mu_jax, model, cov_jax_2d, include_logdet=False)
        lkl_ccl = gaussian_loglike(
            data=cls_rebin_lkl, model=model, cov=cov_rebin_lkl, include_logdet=False)

        assert np.allclose(lkl_jax, lkl_ccl, atol=0, rtol=1e-2)
    print('Cell callculator test passed')


def make_datagen(fiducial_params, powspec_pars_dict):

    datagen = DataGenerator(
        fiducial_params=fiducial_params,
        set_name='TEST PROCEDURES')

    datagen.gen_dNdz(bin_left_edges=powspec_pars_dict['bin_left_edges'],
                     f_fail=powspec_pars_dict['f_fail'],
                     sigma_0=powspec_pars_dict['sigma_0'],
                     xlf=def_agn_xlf,
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


zarr = np.linspace(0.1, 3.5, 751)
dNdz_precomputed = gen_dndz(
    (def_agn_xlf, powspec_pars_dict['slim'], zarr))


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

    assert np.allclose(datagen.bias_arr, def_agn_xlf.b_eff(
        zarr=zarr)[1], atol=0, rtol=1e-4), f'biases b_eff(z) differ!'
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
    error_mag = error_mag
    model_full = np.zeros_like(data_full)
    model_full[data_full > 0] = np.random.normal(
        loc=data_full[data_full > 0], scale=error_mag*data_full[data_full > 0], size=data_full[data_full > 0].shape)
    diff_full = data_full - model_full
    icov_full = dg_full.inv_cov_rebin_lkl
    chi2_full = np.sqrt(
        np.einsum('i, ji,j->', diff_full, icov_full, diff_full))

    model_ignore = model_full[data_full > 0]
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


# test 0: cell calculator function
print('######## STANDARD TESTS ######')
cell_calculator_function_suite()


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
bins_1['bin_left_edges'] = np.array([0.5, 1, 1.2, 1.8, 2.5])
bins_1['sigma_0'] = 0.1

bins_2 = powspec_pars_dict.copy()
bins_2['bin_left_edges'] = np.array([1, 2, 2.5])
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
#               powspec_pars_dict=full_bias) #no works since fisher for bias is deprecated now


print('######## ALL TESTS PASSED #########')
