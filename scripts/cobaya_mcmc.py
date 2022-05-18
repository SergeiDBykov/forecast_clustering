from cobaya import Theory, Likelihood
from .forecast import noise_Cell, sparse_arrays, cov_Cell, gaussian_loglike, DataGenerator, warnings, os
from .luminosity_functions import def_agn_xlf, def_clusters_xlf
import numpy as np



class CCLclustering(Theory):
    """
     a class to return cross and auto angular power spectrum for input cosmological parameters and redshift distributions.

    see https://cobaya.readthedocs.io/en/latest/theories_and_dependencies.html ,CobayaLSS (https://github.com/martinjameswhite/CobayaLSS) and https://github.com/LSSTDESC/DESCBiasChallenge
    """
    # default global class params can be changed
    # power spectrum params
    transfer_function: str = 'boltzmann_camb'
    matter_pk: str = 'linear'
    baryons_pk: str = 'nobaryons'

    # settings for clustering
    sigma_0: float = 0.1
    f_fail: float = 0.1
    slim: float = 1e-14
    l_min: int = 10
    l_max: int = 520
    log_bins: int = 41
    fsky: float = 0.65
    delta_i: int = 3
    use_camb: bool = True
    has_rsd: bool = False
    camb_llimber: int = 110
    use_weighed_bias: bool = False
    density_multiplier: float = 1.
    remove_ignored_cells: bool = True
    type: str = 'AGN'
    bin_left_edges_file: str = 'bin_left_edges.txt'
    fix_cov: bool = False

    # Params it can accept
    params = {'Omega_c': None,
              'Omega_b': None,
              'h': None,
              'n_s': None,
              'sigma8': None}

    def initialize_with_params(self):
        if self.type == 'AGN':
            xlf = def_agn_xlf
        elif self.type == 'Clusters':
            xlf = def_clusters_xlf
        else:
            raise ValueError('type must be AGN or Clusters')
        bin_left_edges = np.loadtxt(self.bin_left_edges_file)
        powspec_pars_dict = {
            'sigma_0': self.sigma_0,
            'f_fail': self.f_fail,
            'slim': self.slim,
            'l_min': self.l_min,
            'l_max': self.l_max,
            'log_bins': self.log_bins,
            'fsky': self.fsky,
            'has_rsd': self.has_rsd,
            'use_weighed_bias': self.use_weighed_bias,
            'density_multiplier': self.density_multiplier,
            'camb_llimber': self.camb_llimber,
            'xlf': xlf,
            'use_camb': self.use_camb,
            'delta_i': self.delta_i,
            'remove_ignored_cells': self.remove_ignored_cells,
            'bin_left_edges': bin_left_edges}

        fiducial_params = {'Omega_c': 0.25, 'Omega_b': 0.05,
                           'h': 0.7, 'sigma8': 0.8, 'n_s': 0.96,
                           'transfer_function': self.transfer_function,
                           'baryons_power_spectrum': self.baryons_pk,
                           'matter_power_spectrum': self.matter_pk, }
        warnings.warn(
            'Using standard cosmological parameters: 0.25, 0.05, 0.7, 0.96,  0.8')
        datagen = DataGenerator(
            fiducial_params=fiducial_params, set_name=self.type,)

        datagen.invoke(
            **powspec_pars_dict, plot_cell=False, plot_dndz=False)
        datagen.invert_cov()
        self.datagen = datagen

    def get_can_provide(self):
        return ['Cell_data_lkl', 'Cell_cov_lkl', 'Cell_inv_cov_lkl']

    def calculate(self, state, want_derived, **params_values_dict):
        datagen = self.datagen
        Omega_c = params_values_dict['Omega_c']
        Omega_b = params_values_dict['Omega_b']
        h = params_values_dict['h']
        n_s = params_values_dict['n_s']
        sigma8 = params_values_dict['sigma8']
        step_params = np.array([Omega_c, Omega_b, h, n_s, sigma8])
        cls_rebin, cls_rebin_lkl, ignored_idx = datagen.Cell_mean(
            step_params)

        if self.fix_cov:
            cov = datagen.cov_rebin_lkl
            icov = datagen.inv_cov_rebin_lkl

        else:

            src_dens_list = self.datagen.tracers_obj.src_dens_list
            noise_power = noise_Cell(src_dens_list)

            cov_rebin = cov_Cell(cls_rebin=cls_rebin, n_logbin=datagen.n_logbin,
                                 ell_rebin=datagen.ell_rebin, noise_power=noise_power, fsky=datagen.fsky,
                                 show_progressbar=False)
            _, cov = sparse_arrays(cls_rebin, cov_rebin, ignored_idx)

            icov = np.linalg.inv(cov)  # type: ignore

        state['Cell_data_lkl'] = cls_rebin_lkl
        state['Cell_cov_lkl'] = cov
        state['Cell_inv_cov_lkl'] = icov


class GaussianClLikelihood(Likelihood):
    data_vector_file: str = None  # type: ignore
    fix_cov: bool = False

    def initialize(self):
        try:
            AllData = np.loadtxt(self.data_vector_file)
        except:
            raise FileNotFoundError(
                f'Data file not found, please check the path, PWD={os.getcwd()}')
        self.data_vector = np.array(AllData)

    def get_requirements(self):
        return ['Cell_data_lkl', 'Cell_inv_cov_lkl', 'Cell_cov_lkl']

    def logp(self, **data_params):

        prov = self.provider
        cls_rebin_lkl_model = prov.get_result('Cell_data_lkl')
        cov_rebin__lkl_theor = prov.get_result('Cell_cov_lkl')
        icov_rebin__lkl_theor = prov.get_result('Cell_inv_cov_lkl')

        assert self.data_vector.shape == cls_rebin_lkl_model.shape, f"data shape {self.data_vector.shape} != cls_rebin shape {cls_rebin_lkl_model.shape}"

        lkl = gaussian_loglike(
            data=self.data_vector, model=cls_rebin_lkl_model, cov=cov_rebin__lkl_theor, icov=icov_rebin__lkl_theor, include_logdet=not self.fix_cov)
        #print(f'include_logdet={not self.fix_cov}')
        if lkl == -np.inf:
            print(
                '$$$$$$$  COBAYA_CLASSES WARNING $$$$$$$$$ problems with lkl (-np.inf). \n  {cosmo_ccl=} \n {cls_rebin=}')
        return lkl

