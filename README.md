# Cosmological forecast for  angular clustering of AGN and Clusters in eROSITA All-Sky X-ray survey


This are  scripts and notebooks which are used to calculate the  results for the cosmological forecast of **eROSITA All-sky survey** in  X-ray. 

The  analysis results I used in my [paper]().
Necessary python packages: `numpy, scipy, matplotlib, seaborn, pandas, numba, numdifftools, tqdm, chainconsumer`; cosmology packages in python: `pyccl, camb, jax_cosmo, cobaya`.

**This  branch is for the code after adressing referee's comments (i.e. before second submission) **

## The structure of the code is as follows:

- `./scripts`  
  the main scripts for the analysis are placed here. It containt a few files to manage the forecast.
    * `./utils.py` is used to set up pathes and contains some utility/plotting functions
    * `./luminosity_funtions.py` contains functions and classes for X-ray  luminosity function (XLF) calulations. It includes XLF for AGN and clusters (the halo model for the latter), calculators of redshift/luminosity  distributions, linear bias factors, and other useful functions. 
    * `./forecast.py` contains classes and functions for the cosmologial calculations, including angular power spectrum and Fisher Matrices.
    * `./tests/` if a folder in which some tests for the code are stored, mainly for comparison with cosmological packages like CAMB, CCL, jax_cosmo.
    * `./k_correction/`  contains a notebook for calulatin K-correction for the cluster spectra and tabulating it in a table `k_correction.npz`.
    

<br>

- `./results` is a folder in which the results are stored, in a form of text/`.npz` files and plots.  

<br>

- `./notebooks`  
  the main notebooks for the analysis are stored here. It containt a few folder and separate notebooks to make the forecast. The instructions are given in each notebook.  The content of the folder is as follows (roughly in the order of  the paper sections):
    * `./luminosity_functions/`  contains three notebooks for the luminosity functions, one for AGN, one for clusters, and one for the plotting both distributions in one figure (in section 3 of the paper). The notebooks demonstrate the use of the classes and functions in `./luminosity_functions.py`, including `dn/dz`, `logN-logS` calculation for AGN and clusters. The instructions are found in the notebooks.
    * `effective_volume.ipynb` calculates the effective volume of the survey for AGN and Cluster tracers (section 2 of the paper). It uses  results from luminosity function calculators. 
    * `./Cell_plot.ipynb` plots the angular power spectrum for AGN and clusters and its derivative for a representative redshift bins (section 4 of the paper). 
    * `./fisher_forecast/BAO/` contains notebooks for the calculation of the BAO significance.
      * `0_BAO_example.ipynb` is the example calculation of the significance in a given photometric setup;
      * `1_BAO_run.py` is a script which iterates over a grid of photometric redshift parameters and saves the resulting BAO SNR in a `.npz` files in the data folder;
      * `2_BAO_plot.ipynb`  is a notebook which plots the BAO SNR as a function of  photometric redshift properties, for AGN and clusters.

  * `./fisher_forecast/cosmo-photoz` contains notebooks for the calculation of the Fisher matrices for AGN and clusters.
    * `0_cosmo_AGN_example.ipynb` is an example of the calculation of Fisher matrices in a given photometric setup for AGN;
    * `1_cosmo_clusters_example.ipynb` is the similar notebook, but for galaxy clusters; 
    * `2_cosmo_run.py` is a script which iterates over a grid of photometric redshift parameters and saves the resulting Fisher matrices in a `.npz` files in the data folder;
    * `3_cosmo_fisher_plots.ipynb`  is a notebook which plots the Fisher matrices with certain  photometric redshift properties, for AGN and clusters, and with priors if needed;
    * `4_cosmo_photoz.ipynb` is a notebook which plots the Fisher matrices' Figure of Merit as a function of  photometric redshift properties, for AGN and clusters;
    * `5_cosmo_tables.ipynb` is a notebook which tabulates the Fisher matrices' Figure of Merit and parameter errors as a function of  photometric redshift properties, for AGN and clusters.
    * `6_survey_pars.ipynb` is a notebook with forecast for different AGN survey parameters.
  * `./mcmc_forecast` contains notebooks in which Monte-Carlo Markov chain are run with `cobaya` cosmological sampler. 
    * `test_cobaya_func.ipynb` is a notebook in which the `cobaya` sampler is tested versus the actual `DataGenerator` object from `scripts.forecast.py`.
    * `AGN_optimistic_h_ns_prior.ipynb` contains the instructins to run mcmc for a Sample of AGN. Data from mcmc is stored in `results/data/mcmc`
  

If you have any questions, please contact me at email indicated in the paper.
