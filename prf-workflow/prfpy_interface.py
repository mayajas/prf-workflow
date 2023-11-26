# prfpy_interface.py

# TODO: 
# - Add docstrings
# - Save output data dict to temp file, then rename to final filename 
# to prevent data loss when slurm job times out

import sys
import os
import pickle
import subprocess
import numpy as np
import nibabel as nib

def pckl_suffix(filename):
        """
        Add .pckl suffix to filename.
        """
        return filename + '.pckl'

class PrfpyStimulus:
    def __init__(self, dir_config, mri_config, prf_config, logger):
        self.dir_config             = dir_config
        self.programs_dir           = dir_config.programs_dir

        self.screen_halfheight_cm   = prf_config.screen_halfheight_cm
        self.screen_distance_cm     = prf_config.screen_distance_cm
        self.TR                     = mri_config.TR

        self.prf_run_config         = mri_config.prf_run_config
        self.prfpy_output_config    = mri_config.prfpy_output_config
        self.output_data_dict_fn    = prf_config.output_data_dict_fn

        self.logger                 = logger
        
        # Create stimulus object
        self.logger.info('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        self.logger.info('Creating stimulus object')
        self.logger.info('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        mri_config.prfpy_output_config  = self._create_stim_obj()
        self.logger.info('Stimulus object created')

    def _create_stim_obj(self):
        """
        Create a stimulus object for prfpy.
        """
        from prfpy.stimulus import PRFStimulus2D

        # First, try loading the output data dict
        if os.path.exists(self.output_data_dict_fn):
            self.logger.info(f'Output data dict already exists: {self.output_data_dict_fn}')
            self.logger.info('Loading output data dict...')
            with open(self.output_data_dict_fn, 'rb') as pickle_file:
                self.prfpy_output_config = pickle.load(pickle_file)
            self.logger.info('Loaded.')
        
        # Define stimulus object for each aperture type
        for aperture_type, config in self.prf_run_config.items():
            self.logger.info(f"[[{aperture_type} aperture]]")

            if not self.prfpy_output_config[aperture_type]['stim']:
                self.prfpy_output_config[aperture_type]['stim'] = \
                                        PRFStimulus2D(screen_size_cm=self.screen_halfheight_cm,
                                            screen_distance_cm=self.screen_distance_cm,
                                            design_matrix=config['design_matrix'],
                                            TR=self.TR)
                ## Save stimulus objects
                with open(self.output_data_dict_fn, 'wb') as pickle_file:
                    pickle.dump(self.prfpy_output_config, pickle_file)
            else:
                self.logger.info('Stimulus object already defined')
            
            
        return self.prfpy_output_config
        

class PrfFitting:
    def __init__(self,dir_config,mri_config,prf_config,project_config,logger):

        # Configuration parameters
        self.dir_config                 = dir_config
        self.prf_config                 = prf_config
        self.prf_run_config             = mri_config.prf_run_config
        self.prfpy_output_config        = mri_config.prfpy_output_config
        self.output_data_dict_fn        = prf_config.output_data_dict_fn

        # General pRF mapping parameters
        self.fit_hrf                    = prf_config.fit_hrf
        self.start_from_avg             = prf_config.start_from_avg
        self.which_model                = prf_config.which_model
        self.fit_css                    = prf_config.fit_css

        # Input parameters of Iso2DGaussianModel
        self.hrf                        = prf_config.hrf
        self.filter_predictions         = prf_config.filter_predictions
        self.filter_type                = prf_config.filter_type

        self.sg_filter_window_length    = prf_config.sg_filter_window_length
        self.sg_filter_polyorder        = prf_config.sg_filter_polyorder
        self.filter_params              = prf_config.filter_params

        self.logger                     = logger

        # Input parameters for grid fit
        self.size_grid, self.ecc_grid, self.polar_grid, self.surround_amplitude_grid, self.surround_size_grid, self.verbose = prf_config.size_grid, prf_config.ecc_grid, prf_config.polar_grid, prf_config.surround_amplitude_grid, prf_config.surround_size_grid, prf_config.verbose

        # Input parameters for iterative fit
        self.rsq_thresh_itfit           = prf_config.rsq_thresh_itfit

        # Number of surfaces
        self.n_surfs                    = project_config.n_surfs

        # Number of cores to use for parallel processing of vertices
        self.n_procs                    = project_config.n_procs

        # Prfpy output filenames
        self.pRF_param_avg_fn           = prf_config.pRF_param_avg_fn
        self.pRF_param_per_depth_fn     = prf_config.pRF_param_per_depth_fn

        # Reference aperture
        self.reference_aperture         = prf_config.reference_aperture

        # Occipital mask
        self.occ_mask_fn                = mri_config.occ_mask_fn

        # maximum eccentricity of stimulus in degrees
        self.max_ecc_deg                = prf_config.max_ecc_deg
        self.meanFunc_mgh_fn            = mri_config.meanFunc_mgh_fn

        # Rsq threshold for excluding vertices in visualization
        self.rsq_thresh_viz             = prf_config.rsq_thresh_viz

        ## PRF fitting 
        self.logger.info('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        self.logger.info('Fitting pRF model')
        self.logger.info('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        self.logger.info('Nr processors: ' + str(self.n_procs))
        self._prf_fitting()

    def _prf_fitting(self):
        """
        Main prf fitting function. 
        This is where all the prfpy stuff is done (except for creating the stimulus object, 
        which was done above in the PrfpyStimulus class).
        """        
        # If only fitting one cortical surface
        if self.n_surfs == 1:
            # first fit iso 2d gaussian
            self.logger.info('%%%%%%%%% Fitting iso 2d gaussian model on single surface %%%%%%%%%')
            self._fit_iso_2d_gaussian(which_surf='single')

            # then extract pRF parameters
            self.logger.info('Extracting pRF parameters for single-layer iso 2d Gaussian model...')
            self._get_prf_params(which_model='Iso',which_surf='single')

            # then fit DoG, if desired
            if self.which_model == 'DoG':
                self.logger.info('%%%%%%%%% Fitting DoG model on single surface %%%%%%%%%')
                self._fit_dog(which_surf='single')

                # then extract pRF parameters
                self.logger.info('Extracting pRF parameters for single-layer iso 2d Gaussian model...')
                self._get_prf_params(which_model='DoG',which_surf='single')

        # If fitting multiple cortical surfaces
        elif self.n_surfs > 1:
            # first fit iso 2d gaussian to avg surface
            self.logger.info('%%%%%%%%% Fitting iso 2d gaussian model on average surface %%%%%%%%%')
            self._fit_iso_2d_gaussian(which_surf='avg')

            # then extract pRF parameters
            self.logger.info('Extracting pRF parameters for avg iso 2d Gaussian model...')
            self._get_prf_params(which_model='Iso',which_surf='avg')

            if self.which_model == 'Iso':
                # then fit iso 2d gaussian to each depth
                self.logger.info('%%%%%%%%% Fitting iso 2d gaussian model across depths %%%%%%%%%')
                self._fit_iso_2d_gaussian(which_surf='depths')

                # then extract pRF parameters
                self.logger.info('Extracting pRF parameters for iso 2d Gaussian model across depths...')
                self._get_prf_params(which_model='Iso',which_surf='depths')

            elif self.which_model == 'DoG':
                # fit DoG to avg surface
                self.logger.info('%%%%%%%%% Fitting DoG model on average surface %%%%%%%%%')
                self.logger.info('On 30 cpus-per-task, 10G mem, with fit_hrf=True this takes about 16 hours...')
                self.logger.info('On 30 cpus-per-task, 10G mem, with fit_hrf=False this takes about 4.81 hours...')
                self._fit_dog(which_surf='avg')

                # then extract pRF parameters
                self.logger.info('Extracting pRF parameters for avg DoG model...')
                self._get_prf_params(which_model='DoG',which_surf='avg')

                # then fit DoG to each depth
                self.logger.info('%%%%%%%%% Fitting DoG model across depths %%%%%%%%%')
                self.logger.info('On 20 cpus-per-task, 10G mem, with fit_hrf=False, this takes about 9.4 hours per surface')
                self.logger.info('On 30 cpus-per-task, 10G mem, with fit_hrf=False, this takes about 6.35 hours per surface')
                self._fit_dog(which_surf='depths')

                # then extract pRF parameters
                self.logger.info('Extracting pRF parameters for DoG model across depths...')
                self._get_prf_params(which_model='DoG',which_surf='depths')
        
    def _fit_iso_2d_gaussian(self,which_surf):
        """
        Define and fit iso 2d gaussian model.
        """      
        from prfpy.model import Iso2DGaussianModel
        from prfpy.fit import Iso2DGaussianFitter, Extend_Iso2DGaussianFitter

        # First, try loading the output data dict
        if os.path.exists(self.output_data_dict_fn):
            self.logger.info(f'Output data dict already exists: {self.output_data_dict_fn}')
            self.logger.info('Loading output data dict...')
            with open(self.output_data_dict_fn, 'rb') as pickle_file:
                self.prfpy_output_config = pickle.load(pickle_file)
            self.logger.info('Loaded.')

        # Define 2D iso Gaussian model and model fitter
        for aperture_type, config in self.prf_run_config.items():
            self.logger.info(f"[[{aperture_type} aperture]]")
           
            if which_surf == 'single' or which_surf == 'avg':
                stimulus    = self.prfpy_output_config[aperture_type]['stim'] 

                # Model definition
                if not self.prfpy_output_config[aperture_type]['is_gg']['avg']:
                    self.logger.info('Defining 2D iso Gaussian model')
                    self.prfpy_output_config[aperture_type]['gg_avg'] = \
                                    Iso2DGaussianModel(stimulus=stimulus,
                                                    filter_predictions=self.filter_predictions,
                                                    filter_type=self.filter_type,
                                                    filter_params=self.filter_params)
                    
                    self.prfpy_output_config[aperture_type]['is_gg']['avg'] = True
                else:
                    self.logger.info('2D iso Gaussian model already defined')

                
                # Defined model fitter and grid fit
                if not self.prfpy_output_config[aperture_type]['is_gf']['avg']['gridfit']:
                    self.logger.info('Defining 2D iso Gaussian model fitter')
                    data        = config['preproc_data_avg']

                    if (aperture_type == self.reference_aperture) or (self.reference_aperture is None) or (self.prfpy_output_config[self.reference_aperture]['is_gf']['avg']['itfit'] == False):
                        self.prfpy_output_config[aperture_type]['gf_avg'] = \
                                        Iso2DGaussianFitter(data=data, 
                                                            model=self.prfpy_output_config[aperture_type]['gg_avg'], 
                                                            n_jobs=self.n_procs, fit_hrf=self.fit_hrf)
                    else:
                        self.logger.info('Using reference aperture ('+self.reference_aperture+') to initiate '+aperture_type+' aperture fitting.')
                        self.prfpy_output_config[aperture_type]['gf_avg'] = \
                                        Extend_Iso2DGaussianFitter(data=data, 
                                                            model=self.prfpy_output_config[aperture_type]['gg_avg'], 
                                                            previous_gaussian_fitter=self.prfpy_output_config[self.reference_aperture]['gf_avg'],
                                                            n_jobs=self.n_procs, fit_hrf=self.fit_hrf)


                    # Fit model: 
                    # grid fit
                    self.logger.info('Grid fit')
                    try:
                        self.prfpy_output_config[aperture_type]['gf_avg'].grid_fit(ecc_grid=self.ecc_grid,
                                                                                    polar_grid=self.polar_grid,
                                                                                    size_grid=self.size_grid,
                                                                                    verbose=self.verbose,
                                                                                    n_batches=self.n_procs)
                        self.prfpy_output_config[aperture_type]['is_gf']['avg']['gridfit'] = True

                        # Save output
                        self.logger.info('Saving output...')
                        with open(self.output_data_dict_fn, 'wb') as pickle_file:
                            pickle.dump(self.prfpy_output_config, pickle_file)
                    except subprocess.CalledProcessError as e:
                        self.logger.error(f"Grid fit failed with return code {e.returncode}: {e}")
                        self.prfpy_output_config[aperture_type]['gf_avg']       = []
                        self.prfpy_output_config[aperture_type]['is_gf']['avg']['gridfit'] = False

                        # Save output
                        self.logger.info('Saving output...')
                        with open(self.output_data_dict_fn, 'wb') as pickle_file:
                            pickle.dump(self.prfpy_output_config, pickle_file)

                        # Exit with a non-zero code to indicate error
                        sys.exit(1)
                else: 
                    self.logger.info('Grid fit already performed')

                # Iterative fit
                if not self.prfpy_output_config[aperture_type]['is_gf']['avg']['itfit']:
                    self.logger.info('Iterative fit')
                    try:
                        self.prfpy_output_config[aperture_type]['gf_avg'].iterative_fit(rsq_threshold=self.rsq_thresh_itfit, verbose=self.verbose)
                        self.prfpy_output_config[aperture_type]['is_gf']['avg']['itfit'] = True

                        # Save output
                        self.logger.info('Saving output...')
                        with open(self.output_data_dict_fn, 'wb') as pickle_file:
                            pickle.dump(self.prfpy_output_config, pickle_file)
                    except  subprocess.CalledProcessError as e:
                        self.logger.error(f"Iterative fit failed with return code {e.returncode}: {e}")
                        self.prfpy_output_config[aperture_type]['is_gf']['avg']['itfit'] = False

                        # Save output
                        self.logger.info('Saving output...')
                        with open(self.output_data_dict_fn, 'wb') as pickle_file:
                            pickle.dump(self.prfpy_output_config, pickle_file)

                        # Exit with a non-zero code to indicate error
                        sys.exit(1)
                else:
                    self.logger.info('Iterative fit already performed')

            elif which_surf == 'depths':
                for depth in range(0,self.n_surfs):
                    self.logger.info(f"[Depth {depth}]")

                    # Defined model fitter and grid fit
                    if not self.prfpy_output_config[aperture_type]['is_gf']['per_depth']['gridfit'][depth]:
                        self.logger.info('Defining 2D iso Gaussian model fitter')
                    
                        data        = config['preproc_data_per_depth'][depth]
                        if self.start_from_avg:
                            self.prfpy_output_config[aperture_type]['gf_per_depth'][depth] = \
                                            Extend_Iso2DGaussianFitter(data=data, 
                                                                model=self.prfpy_output_config[aperture_type]['gg_avg'], 
                                                                previous_gaussian_fitter=self.prfpy_output_config[aperture_type]['gf_avg'],
                                                                n_jobs=self.n_procs, fit_hrf=self.fit_hrf)
                        else:
                            if (aperture_type == self.reference_aperture) or (self.reference_aperture is None) or (self.prfpy_output_config[self.reference_aperture]['is_gf']['per_depth']['itfit'][depth] == False):
                                self.prfpy_output_config[aperture_type]['gf_per_depth'][depth] = \
                                                Iso2DGaussianFitter(data=data, 
                                                                    model=self.prfpy_output_config[aperture_type]['gg_avg'], 
                                                                    n_jobs=self.n_procs, fit_hrf=self.fit_hrf)
                            else:
                                self.logger.info('Using reference aperture ('+self.reference_aperture+') to initiate '+aperture_type+' aperture fitting.')
                                self.prfpy_output_config[aperture_type]['gf_per_depth'][depth] = \
                                            Extend_Iso2DGaussianFitter(data=data, 
                                                                model=self.prfpy_output_config[aperture_type]['gg_avg'], 
                                                                previous_gaussian_fitter=self.prfpy_output_config[self.reference_aperture]['gf_per_depth'][depth],
                                                                n_jobs=self.n_procs, fit_hrf=self.fit_hrf)
                        
                        # Fit model: grid fit
                        self.logger.info('Grid fit')
                        try:
                            self.prfpy_output_config[aperture_type]['gf_per_depth'][depth].grid_fit(ecc_grid=self.ecc_grid,
                                                                                                    polar_grid=self.polar_grid,
                                                                                                    size_grid=self.size_grid,
                                                                                                    verbose=self.verbose,
                                                                                                    n_batches=self.n_procs)
                            self.prfpy_output_config[aperture_type]['is_gf']['per_depth']['gridfit'][depth] = True
                        
                            # Save output
                            self.logger.info('Saving output...')
                            with open(self.output_data_dict_fn, 'wb') as pickle_file:
                                pickle.dump(self.prfpy_output_config, pickle_file)
                        except subprocess.CalledProcessError as e:
                            self.logger.error(f"Grid fit failed with return code {e.returncode}: {e}")
                            self.prfpy_output_config[aperture_type]['gf_per_depth'][depth] = []
                            self.prfpy_output_config[aperture_type]['is_gf']['per_depth']['gridfit'][depth] = False

                            # Save output
                            self.logger.info('Saving output...')
                            with open(self.output_data_dict_fn, 'wb') as pickle_file:
                                pickle.dump(self.prfpy_output_config, pickle_file)

                            # Exit with a non-zero code to indicate error
                            sys.exit(1)
                    else:
                        self.logger.info('Grid fit already performed on depth ' + str(depth))
                    
                    # Iterative fit
                    if not self.prfpy_output_config[aperture_type]['is_gf']['per_depth']['itfit'][depth]:
                        try:
                            self.logger.info('Iterative fit')
                            self.prfpy_output_config[aperture_type]['gf_per_depth'][depth].iterative_fit(rsq_threshold=self.rsq_thresh_itfit, verbose=self.verbose)
                            self.prfpy_output_config[aperture_type]['is_gf']['per_depth']['itfit'][depth] = True

                            # Save output
                            self.logger.info('Saving output...')
                            with open(self.output_data_dict_fn, 'wb') as pickle_file:
                                pickle.dump(self.prfpy_output_config, pickle_file)
                        except  subprocess.CalledProcessError as e:
                            self.logger.error(f"Iterative fit failed with return code {e.returncode}: {e}")
                            self.prfpy_output_config[aperture_type]['is_gf']['per_depth']['itfit'][depth] = False

                            # Save output
                            self.logger.info('Saving output...')
                            with open(self.output_data_dict_fn, 'wb') as pickle_file:
                                pickle.dump(self.prfpy_output_config, pickle_file)

                            # Exit with a non-zero code to indicate error
                            sys.exit(1)
                    else:
                        self.logger.info('Iterative fit already performed on depth ' + str(depth))

    def _fit_dog(self,which_surf):
        """
        Define and fit DoG model.
        """
        from prfpy.model import DoG_Iso2DGaussianModel
        from prfpy.fit import DoG_Iso2DGaussianFitter

        # First, try loading the output data dict
        if os.path.exists(self.output_data_dict_fn):
            self.logger.info(f'Output data dict already exists: {self.output_data_dict_fn}')
            self.logger.info('Loading output data dict...')
            with open(self.output_data_dict_fn, 'rb') as pickle_file:
                self.prfpy_output_config = pickle.load(pickle_file)
            self.logger.info('Loaded.')

        # Define DoG model and model fitter
        for aperture_type, config in self.prf_run_config.items():
            self.logger.info(f"[[{aperture_type} aperture]]")
            
            if which_surf == 'single' or which_surf == 'avg':
                stimulus    = self.prfpy_output_config[aperture_type]['stim'] 

                # Model definition
                if not self.prfpy_output_config[aperture_type]['is_gg']['dog_avg']:
                    # Define DoG model
                    self.logger.info('Defining DoG model')
                    self.prfpy_output_config[aperture_type]['gg_dog_avg'] = \
                                    DoG_Iso2DGaussianModel(stimulus=stimulus,
                                                    filter_predictions=self.filter_predictions,
                                                    filter_type=self.filter_type,
                                                    filter_params=self.filter_params)
                    
                    self.prfpy_output_config[aperture_type]['is_gg']['dog_avg'] = True
                else:
                    self.logger.info('DoG model already defined')


                # Model fit and grid fit
                if not self.prfpy_output_config[aperture_type]['is_gf']['dog_avg']['gridfit']:
                    # Defined model fitter
                    self.logger.info('Defining DoG model fitter')
                    data                    = config['preproc_data_avg']
                    previous_gaussian_fitter=self.prfpy_output_config[aperture_type]['gf_avg']

                    self.prfpy_output_config[aperture_type]['gf_dog_avg'] = \
                                    DoG_Iso2DGaussianFitter(data=data, 
                                                        model=self.prfpy_output_config[aperture_type]['gg_dog_avg'], 
                                                        n_jobs=self.n_procs, fit_hrf=self.fit_hrf, fit_css=self.fit_css,
                                                        previous_gaussian_fitter=previous_gaussian_fitter)
                    # Fit model: 
                    # grid fit
                    self.logger.info('Grid fit')
                    try:
                        self.prfpy_output_config[aperture_type]['gf_dog_avg'].grid_fit(surround_amplitude_grid = self.surround_amplitude_grid,
                                                                                        surround_size_grid = self.surround_size_grid,
                                                                                        rsq_threshold=self.rsq_thresh_itfit, 
                                                                                        verbose=self.verbose, 
                                                                                        gaussian_params=previous_gaussian_fitter.gridsearch_params)
                        self.prfpy_output_config[aperture_type]['is_gf']['dog_avg']['gridfit'] = True

                        # Save output
                        self.logger.info('Saving output...')
                        with open(self.output_data_dict_fn, 'wb') as pickle_file:
                            pickle.dump(self.prfpy_output_config, pickle_file)
                    except subprocess.CalledProcessError as e:
                        self.logger.error(f"Grid fit failed with return code {e.returncode}: {e}")
                        self.prfpy_output_config[aperture_type]['gf_avg']       = []
                        self.prfpy_output_config[aperture_type]['is_gf']['dog_avg']['gridfit'] = False

                        # Save output
                        self.logger.info('Saving output...')
                        with open(self.output_data_dict_fn, 'wb') as pickle_file:
                            pickle.dump(self.prfpy_output_config, pickle_file)

                        # Exit with a non-zero code to indicate error
                        sys.exit(1)
                else: 
                    self.logger.info('Grid fit already performed')


                # Iterative fit
                if not self.prfpy_output_config[aperture_type]['is_gf']['dog_avg']['itfit']:
                    self.logger.info('Iterative fit')
                    try:
                        self.prfpy_output_config[aperture_type]['gf_dog_avg'].iterative_fit(rsq_threshold=self.rsq_thresh_itfit, verbose=self.verbose)
                        self.prfpy_output_config[aperture_type]['is_gf']['dog_avg']['itfit'] = True

                        # Save output
                        self.logger.info('Saving output...')
                        with open(self.output_data_dict_fn, 'wb') as pickle_file:
                            pickle.dump(self.prfpy_output_config, pickle_file)
                    except  subprocess.CalledProcessError as e:
                        self.logger.error(f"Iterative fit failed with return code {e.returncode}: {e}")
                        self.prfpy_output_config[aperture_type]['is_gf']['dog_avg']['itfit'] = False

                        # Save output
                        self.logger.info('Saving output...')
                        with open(self.output_data_dict_fn, 'wb') as pickle_file:
                            pickle.dump(self.prfpy_output_config, pickle_file)

                        # Exit with a non-zero code to indicate error
                        sys.exit(1)
                else:
                    self.logger.info('Iterative fit already performed')

                
            elif which_surf == 'depths':
                for depth in range(0,self.n_surfs):
                    self.logger.info(f"[Depth {depth}]")

                    # Defined model fitter and grid fit
                    if not self.prfpy_output_config[aperture_type]['is_gf']['dog_per_depth']['gridfit'][depth]:
                        data        = config['preproc_data_per_depth'][depth]
                        previous_gaussian_fitter    = self.prfpy_output_config[aperture_type]['gf_dog_avg']

                        self.prfpy_output_config[aperture_type]['gf_dog_per_depth'][depth] = \
                                            DoG_Iso2DGaussianFitter(data=data, 
                                                        model=self.prfpy_output_config[aperture_type]['gg_dog_avg'], 
                                                        n_jobs=self.n_procs, fit_hrf=self.fit_hrf, fit_css=self.fit_css,
                                                        previous_gaussian_fitter=previous_gaussian_fitter)
                        # Fit model: grid fit
                        self.logger.info('Grid fit')
                        try:
                            self.prfpy_output_config[aperture_type]['gf_dog_per_depth'][depth].grid_fit(surround_amplitude_grid = self.surround_amplitude_grid,
                                                                               surround_size_grid = self.surround_size_grid,
                                                                               rsq_threshold=self.rsq_thresh_itfit, 
                                                                               verbose=self.verbose, 
                                                                               gaussian_params=previous_gaussian_fitter.gridsearch_params)
                            self.prfpy_output_config[aperture_type]['is_gf']['dog_per_depth']['gridfit'][depth] = True
                        
                            # Save output
                            self.logger.info('Saving output...')
                            with open(self.output_data_dict_fn, 'wb') as pickle_file:
                                pickle.dump(self.prfpy_output_config, pickle_file)
                        except subprocess.CalledProcessError as e:
                            self.logger.error(f"Grid fit failed with return code {e.returncode}: {e}")
                            self.prfpy_output_config[aperture_type]['gf_dog_per_depth'][depth] = []
                            self.prfpy_output_config[aperture_type]['is_gf']['dog_per_depth']['gridfit'][depth] = False

                            # Save output
                            self.logger.info('Saving output...')
                            with open(self.output_data_dict_fn, 'wb') as pickle_file:
                                pickle.dump(self.prfpy_output_config, pickle_file)

                            # Exit with a non-zero code to indicate error
                            sys.exit(1)
                    else:
                        self.logger.info('Grid fit already performed on depth ' + str(depth))

                    # Iterative fit
                    if not self.prfpy_output_config[aperture_type]['is_gf']['dog_per_depth']['itfit'][depth]:
                        try:
                            self.logger.info('Iterative fit')
                            self.prfpy_output_config[aperture_type]['gf_dog_per_depth'][depth].iterative_fit(rsq_threshold=self.rsq_thresh_itfit, verbose=self.verbose)
                            self.prfpy_output_config[aperture_type]['is_gf']['dog_per_depth']['itfit'][depth] = True

                            # Save output
                            self.logger.info('Saving output...')
                            with open(self.output_data_dict_fn, 'wb') as pickle_file:
                                pickle.dump(self.prfpy_output_config, pickle_file)
                        except  subprocess.CalledProcessError as e:
                            self.logger.error(f"Iterative fit failed with return code {e.returncode}: {e}")
                            self.prfpy_output_config[aperture_type]['is_gf']['dog_per_depth']['itfit'][depth] = False

                            # Save output
                            self.logger.info('Saving output...')
                            with open(self.output_data_dict_fn, 'wb') as pickle_file:
                                pickle.dump(self.prfpy_output_config, pickle_file)

                            # Exit with a non-zero code to indicate error
                            sys.exit(1)
                    else:
                        self.logger.info('Iterative fit already performed on depth ' + str(depth))         


    def _get_prf_param_fn(self,which_surf,which_model):

        if which_surf == 'single' or which_surf == 'avg':
            if which_model == 'Iso':
                pRF_param_fn = pckl_suffix(self.pRF_param_avg_fn)
            elif which_model == 'DoG':
                pRF_param_fn = pckl_suffix(self.pRF_param_avg_fn+'_DoG')
        elif which_surf == 'depths':
            if which_model == 'Iso':
                pRF_param_fn = pckl_suffix(self.pRF_param_per_depth_fn)
            elif which_model == 'DoG':
                pRF_param_fn = pckl_suffix(self.pRF_param_per_depth_fn+'_DoG')

        return pRF_param_fn


    def _get_prf_params(self,which_model,which_surf):
        """
        Extract pRF parameters from model fitter.
        """""
        ## PRF parameter estimates
        pRF_param_fn = self._get_prf_param_fn(which_surf,which_model)
        
        # Extract pRF parameter estimates from iterative fit result
        if which_surf == 'single' or which_surf == 'avg':
            # For single and average surfaces
            if not os.path.exists(pRF_param_fn):
                self.logger.info('{} does not yet exist'.format(pRF_param_fn))
                # Initialize pRF parameters 
                prf_params      = {
                    key:{
                    'x': [],
                    'y': [],
                    'prf_size': [],
                    'prf_amp': [],
                    'bold_baseline': [],
                    'srf_amp': [],
                    'srf_size': [],
                    'hrf_1': [],
                    'hrf_2': [],
                    'total_rsq': [],
                    'polar': [],
                    'ecc': []
                    } for key in self.prfpy_output_config
                }
                
                for aperture_type, config in self.prf_run_config.items():
                    self.logger.info(f"[[{aperture_type} aperture]]")
                    if which_model == 'Iso':
                        prf_params[aperture_type]['x']=self.prfpy_output_config[aperture_type]['gf_avg'].iterative_search_params[:,0]
                        prf_params[aperture_type]['y']=self.prfpy_output_config[aperture_type]['gf_avg'].iterative_search_params[:,1]
                        prf_params[aperture_type]['prf_size']=self.prfpy_output_config[aperture_type]['gf_avg'].iterative_search_params[:,2]
                        prf_params[aperture_type]['prf_amp']=self.prfpy_output_config[aperture_type]['gf_avg'].iterative_search_params[:,3]
                        prf_params[aperture_type]['bold_baseline']=self.prfpy_output_config[aperture_type]['gf_avg'].iterative_search_params[:,4]
                        prf_params[aperture_type]['srf_amp']=np.empty_like(prf_params[aperture_type]['x']).fill(np.nan)
                        prf_params[aperture_type]['srf_size']=np.empty_like(prf_params[aperture_type]['x']).fill(np.nan)

                        if self.fit_hrf:
                            prf_params[aperture_type]['hrf_1']=self.prfpy_output_config[aperture_type]['gf_avg'].iterative_search_params[:,5]
                            prf_params[aperture_type]['hrf_2']=self.prfpy_output_config[aperture_type]['gf_avg'].iterative_search_params[:,6]
                        else:
                            prf_params[aperture_type]['hrf_1']=np.empty_like(prf_params[aperture_type]['x']).fill(np.nan)
                            prf_params[aperture_type]['hrf_2']=np.empty_like(prf_params[aperture_type]['x']).fill(np.nan)

                        prf_params[aperture_type]['total_rsq']=self.prfpy_output_config[aperture_type]['gf_avg'].iterative_search_params[:,-1]

                        #Calculate polar angle and eccentricity maps
                        prf_params[aperture_type]['polar']=np.angle(prf_params[aperture_type]['x'] + 1j*prf_params[aperture_type]['y'])
                        prf_params[aperture_type]['ecc']=np.abs(prf_params[aperture_type]['x'] + 1j*prf_params[aperture_type]['y'])

                    elif which_model == 'DoG':
                        prf_params[aperture_type]['x']=self.prfpy_output_config[aperture_type]['gf_dog_avg'].iterative_search_params[:,0]
                        prf_params[aperture_type]['y']=self.prfpy_output_config[aperture_type]['gf_dog_avg'].iterative_search_params[:,1]
                        prf_params[aperture_type]['prf_size']=self.prfpy_output_config[aperture_type]['gf_dog_avg'].iterative_search_params[:,2]
                        prf_params[aperture_type]['prf_amp']=self.prfpy_output_config[aperture_type]['gf_dog_avg'].iterative_search_params[:,3]
                        prf_params[aperture_type]['bold_baseline']=self.prfpy_output_config[aperture_type]['gf_dog_avg'].iterative_search_params[:,4]
                        prf_params[aperture_type]['srf_amp']=self.prfpy_output_config[aperture_type]['gf_dog_avg'].iterative_search_params[:,5]
                        prf_params[aperture_type]['srf_size']=self.prfpy_output_config[aperture_type]['gf_dog_avg'].iterative_search_params[:,6]

                        if self.fit_hrf:
                            # TODO: make sure these are the correct indices!!!
                            prf_params[aperture_type]['hrf_1']=self.prfpy_output_config[aperture_type]['gf_dog_avg'].iterative_search_params[:,7]
                            prf_params[aperture_type]['hrf_2']=self.prfpy_output_config[aperture_type]['gf_dog_avg'].iterative_search_params[:,8]
                        else:
                            prf_params[aperture_type]['hrf_1']=np.empty_like(prf_params[aperture_type]['x']).fill(np.nan)
                            prf_params[aperture_type]['hrf_2']=np.empty_like(prf_params[aperture_type]['x']).fill(np.nan)
                        
                        prf_params[aperture_type]['total_rsq']=self.prfpy_output_config[aperture_type]['gf_dog_avg'].iterative_search_params[:,-1]

                        #Calculate polar angle and eccentricity maps
                        prf_params[aperture_type]['polar']=np.angle(prf_params[aperture_type]['x'] + 1j*prf_params[aperture_type]['y'])
                        prf_params[aperture_type]['ecc']=np.abs(prf_params[aperture_type]['x'] + 1j*prf_params[aperture_type]['y'])

                # Save pRF parameters
                with open(pRF_param_fn, 'wb') as pickle_file:
                    pickle.dump(prf_params, pickle_file)

            elif os.path.exists(pRF_param_fn):
                with open(pRF_param_fn,'rb') as pickle_file:
                    prf_params = pickle.load(pickle_file)

        elif which_surf == 'depths':
            # For individual surfaces
            if not os.path.exists(pRF_param_fn):
                self.logger.info('{} does not yet exist'.format(pRF_param_fn))

                # Initialize pRF parameters
                prf_params      = {
                    key:{
                    'x': [0] * self.n_surfs,
                    'y': [0] * self.n_surfs,
                    'prf_size': [0] * self.n_surfs,
                    'prf_amp': [0] * self.n_surfs,
                    'bold_baseline': [0] * self.n_surfs,
                    'srf_amp': [0] * self.n_surfs,
                    'srf_size': [0] * self.n_surfs,
                    'hrf_1': [0] * self.n_surfs,
                    'hrf_2': [0] * self.n_surfs,
                    'total_rsq': [0] * self.n_surfs,
                    'polar': [0] * self.n_surfs,
                    'ecc': [0] * self.n_surfs
                    } for key in self.prfpy_output_config
                }

                for aperture_type, config in self.prf_run_config.items():
                    self.logger.info(f"[[{aperture_type} aperture]]")
                    
                    for depth in range(0,self.n_surfs):
                        if which_model == 'Iso':
                            prf_params[aperture_type]['x'][depth]                = self.prfpy_output_config[aperture_type]['gf_per_depth'][depth].iterative_search_params[:,0]
                            prf_params[aperture_type]['y'][depth]                = self.prfpy_output_config[aperture_type]['gf_per_depth'][depth].iterative_search_params[:,1]
                            prf_params[aperture_type]['prf_size'][depth]         = self.prfpy_output_config[aperture_type]['gf_per_depth'][depth].iterative_search_params[:,2]
                            prf_params[aperture_type]['prf_amp'][depth]          = np.empty_like(prf_params[aperture_type]['x'][depth]).fill(np.nan)
                            prf_params[aperture_type]['bold_baseline'][depth]    = np.empty_like(prf_params[aperture_type]['x'][depth]).fill(np.nan)
                            prf_params[aperture_type]['srf_amp'][depth]          = np.empty_like(prf_params[aperture_type]['x'][depth]).fill(np.nan)
                            prf_params[aperture_type]['srf_size'][depth]         = np.empty_like(prf_params[aperture_type]['x'][depth]).fill(np.nan)

                            if self.fit_hrf:
                                prf_params[aperture_type]['hrf_1'][depth]        = self.prfpy_output_config[aperture_type]['gf_per_depth'][depth].iterative_search_params[:,5]
                                prf_params[aperture_type]['hrf_2'][depth]        = self.prfpy_output_config[aperture_type]['gf_per_depth'][depth].iterative_search_params[:,6]
                            else:
                                prf_params[aperture_type]['hrf_1'][depth]        = np.empty_like(prf_params[aperture_type]['x'][depth]).fill(np.nan)
                                prf_params[aperture_type]['hrf_2'][depth]        = np.empty_like(prf_params[aperture_type]['x'][depth]).fill(np.nan)

                            prf_params[aperture_type]['total_rsq'][depth]        = self.prfpy_output_config[aperture_type]['gf_per_depth'][depth].iterative_search_params[:,-1]

                            # Calculate polar angle and eccentricity maps
                            prf_params[aperture_type]['polar'][depth]            = np.angle(prf_params[aperture_type]['x'][depth] + 1j*prf_params[aperture_type]['y'][depth])
                            prf_params[aperture_type]['ecc'][depth]              = np.abs(prf_params[aperture_type]['x'][depth] + 1j*prf_params[aperture_type]['y'][depth])

                        elif which_model == 'DoG':
                            prf_params[aperture_type]['x'][depth]                = self.prfpy_output_config[aperture_type]['gf_dog_per_depth'][depth].iterative_search_params[:,0]
                            prf_params[aperture_type]['y'][depth]                = self.prfpy_output_config[aperture_type]['gf_dog_per_depth'][depth].iterative_search_params[:,1]
                            prf_params[aperture_type]['prf_size'][depth]         = self.prfpy_output_config[aperture_type]['gf_dog_per_depth'][depth].iterative_search_params[:,2]
                            prf_params[aperture_type]['prf_amp'][depth]          = self.prfpy_output_config[aperture_type]['gf_dog_per_depth'][depth].iterative_search_params[:,3]
                            prf_params[aperture_type]['bold_baseline'][depth]    = self.prfpy_output_config[aperture_type]['gf_dog_per_depth'][depth].iterative_search_params[:,4]
                            prf_params[aperture_type]['srf_amp'][depth]          = self.prfpy_output_config[aperture_type]['gf_dog_per_depth'][depth].iterative_search_params[:,5]
                            prf_params[aperture_type]['srf_size'][depth]         = self.prfpy_output_config[aperture_type]['gf_dog_per_depth'][depth].iterative_search_params[:,6]

                            if self.fit_hrf:
                                prf_params[aperture_type]['hrf_1'][depth]        = self.prfpy_output_config[aperture_type]['gf_dog_per_depth'][depth].iterative_search_params[:,7]
                                prf_params[aperture_type]['hrf_2'][depth]        = self.prfpy_output_config[aperture_type]['gf_dog_per_depth'][depth].iterative_search_params[:,8]
                            else:
                                prf_params[aperture_type]['hrf_1'][depth]        = np.empty_like(prf_params[aperture_type]['x'][depth]).fill(np.nan)
                                prf_params[aperture_type]['hrf_2'][depth]        = np.empty_like(prf_params[aperture_type]['x'][depth]).fill(np.nan)

                            prf_params[aperture_type]['total_rsq'][depth]        = self.prfpy_output_config[aperture_type]['gf_dog_per_depth'][depth].iterative_search_params[:,-1]

                            # Calculate polar angle and eccentricity maps
                            prf_params[aperture_type]['polar'][depth]            = np.angle(prf_params[aperture_type]['x'][depth] + 1j*prf_params[aperture_type]['y'][depth])
                            prf_params[aperture_type]['ecc'][depth]              = np.abs(prf_params[aperture_type]['x'][depth] + 1j*prf_params[aperture_type]['y'][depth])

                # Save pRF parameters
                with open(pRF_param_fn, 'wb') as pickle_file:
                    pickle.dump(prf_params, pickle_file)
                    
            elif os.path.exists(pRF_param_fn):
                with open(pRF_param_fn,'rb') as pickle_file:
                    prf_params = pickle.load(pickle_file)
            

        ###########################################################################################
        # Save pRF parameters to mgh for delineations and visualization
        with open(self.occ_mask_fn, 'rb') as pickle_file:
                occ_mask,n_vtx = pickle.load(pickle_file)

        if which_surf == 'single' or which_surf == 'avg':
            
            # Unmask avg pRF parameters
            unmask_x               = np.zeros(n_vtx)
            unmask_y               = np.zeros(n_vtx)
            unmask_prf_size        = np.zeros(n_vtx)
            unmask_prf_amp         = np.zeros(n_vtx)
            unmask_bold_baseline   = np.zeros(n_vtx)
            unmask_srf_amp         = np.zeros(n_vtx)
            unmask_srf_size        = np.zeros(n_vtx)
            unmask_hrf_1           = np.zeros(n_vtx)
            unmask_hrf_2           = np.zeros(n_vtx)
            unmask_rsq             = np.zeros(n_vtx)
            unmask_polar           = np.zeros(n_vtx)
            unmask_ecc             = np.zeros(n_vtx)
            

            unmask_x[occ_mask]              = prf_params['bar']['x']
            unmask_y[occ_mask]              = prf_params['bar']['y']
            unmask_prf_size[occ_mask]       = prf_params['bar']['prf_size']
            unmask_prf_amp[occ_mask]        = prf_params['bar']['prf_amp']
            unmask_bold_baseline[occ_mask]  = prf_params['bar']['bold_baseline']
            unmask_srf_amp[occ_mask]        = prf_params['bar']['srf_amp']
            unmask_srf_size[occ_mask]       = prf_params['bar']['srf_size']
            unmask_hrf_1[occ_mask]          = prf_params['bar']['hrf_1']
            unmask_hrf_2[occ_mask]          = prf_params['bar']['hrf_2']
            unmask_rsq[occ_mask]            = prf_params['bar']['total_rsq']
            unmask_polar[occ_mask]          = prf_params['bar']['polar']
            unmask_ecc[occ_mask]            = prf_params['bar']['ecc']

            # Constrain prf maps to realistic eccentricities & pRF sizes
            max_ecc_deg = self.max_ecc_deg
            pRF_thresh  = self.max_ecc_deg   
            rsq_thresh  = self.rsq_thresh_viz

            # remove vertices where eccentricity is larger than max stimulus ecc
            # remove vertices where pRF size is negative
            # remove vertices where pRF size is greater than max_ecc_deg
            # remove vertices where rsq is less than rsq_thresh
            condition = (unmask_ecc>max_ecc_deg) | (unmask_prf_size<0) | (unmask_prf_size>pRF_thresh) | (unmask_rsq < rsq_thresh)
            unmask_x[condition]                = np.nan
            unmask_y[condition]                = np.nan
            unmask_prf_size[condition]         = np.nan
            unmask_prf_amp[condition]          = np.nan
            unmask_bold_baseline[condition]    = np.nan
            unmask_srf_amp[condition]          = np.nan
            unmask_srf_size[condition]         = np.nan
            unmask_hrf_1[condition]            = np.nan
            unmask_hrf_2[condition]            = np.nan
            unmask_rsq[condition]              = np.nan
            unmask_polar[condition]            = np.nan
            unmask_ecc[condition]              = np.nan
            
            # set nans to 0
            unmask_x[np.isnan(unmask_x)]         = 0.
            unmask_y[np.isnan(unmask_y)]         = 0.
            unmask_prf_size[np.isnan(unmask_prf_size)] = 0.
            unmask_prf_amp[np.isnan(unmask_prf_amp)] = 0.
            unmask_bold_baseline[np.isnan(unmask_bold_baseline)] = 0.
            unmask_srf_amp[np.isnan(unmask_srf_amp)] = 0.
            unmask_srf_size[np.isnan(unmask_srf_size)] = 0.
            unmask_hrf_1[np.isnan(unmask_hrf_1)] = 0.
            unmask_hrf_2[np.isnan(unmask_hrf_2)] = 0.
            unmask_rsq[np.isnan(unmask_rsq)]     = 0.
            unmask_polar[np.isnan(unmask_polar)] = 0.
            unmask_ecc[np.isnan(unmask_ecc)]     = 0.

            # Save maps to .mgh files for manual delineations
            meanFunc_mgh_nib = nib.freesurfer.mghformat.load(self.meanFunc_mgh_fn)
            affine = meanFunc_mgh_nib.affine

            if not os.path.exists(self.prf_config.x_map_mgh):
                nib.save(nib.freesurfer.mghformat.MGHImage(unmask_x.astype(np.float32, order = "C"),affine=affine),self.prf_config.x_map_mgh)
            if not os.path.exists(self.prf_config.y_map_mgh):
                nib.save(nib.freesurfer.mghformat.MGHImage(unmask_y.astype(np.float32, order = "C"),affine=affine),self.prf_config.y_map_mgh)
            if not os.path.exists(self.prf_config.prf_size_map_mgh):
                nib.save(nib.freesurfer.mghformat.MGHImage(unmask_prf_size.astype(np.float32, order = "C"),affine=affine),self.prf_config.prf_size_map_mgh)
            if not os.path.exists(self.prf_config.prf_amp_map_mgh):
                nib.save(nib.freesurfer.mghformat.MGHImage(unmask_prf_amp.astype(np.float32, order = "C"),affine=affine),self.prf_config.prf_amp_map_mgh)
            if not os.path.exists(self.prf_config.bold_baseline_map_mgh):
                nib.save(nib.freesurfer.mghformat.MGHImage(unmask_bold_baseline.astype(np.float32, order = "C"),affine=affine),self.prf_config.bold_baseline_map_mgh)
            if not os.path.exists(self.prf_config.srf_amp_map_mgh):
                nib.save(nib.freesurfer.mghformat.MGHImage(unmask_srf_amp.astype(np.float32, order = "C"),affine=affine),self.prf_config.srf_amp_map_mgh)
            if not os.path.exists(self.prf_config.srf_size_map_mgh):
                nib.save(nib.freesurfer.mghformat.MGHImage(unmask_srf_size.astype(np.float32, order = "C"),affine=affine),self.prf_config.srf_size_map_mgh)
            if not os.path.exists(self.prf_config.hrf_1_map_mgh):
                nib.save(nib.freesurfer.mghformat.MGHImage(unmask_hrf_1.astype(np.float32, order = "C"),affine=affine),self.prf_config.hrf_1_map_mgh)
            if not os.path.exists(self.prf_config.hrf_2_map_mgh):
                nib.save(nib.freesurfer.mghformat.MGHImage(unmask_hrf_2.astype(np.float32, order = "C"),affine=affine),self.prf_config.hrf_2_map_mgh)
            if not os.path.exists(self.prf_config.polar_map_mgh):
                nib.save(nib.freesurfer.mghformat.MGHImage(unmask_polar.astype(np.float32, order = "C"),affine=affine),self.prf_config.polar_map_mgh)
            if not os.path.exists(self.prf_config.ecc_map_mgh):
                nib.save(nib.freesurfer.mghformat.MGHImage(unmask_ecc.astype(np.float32, order = "C"),affine=affine),self.prf_config.ecc_map_mgh)
            if not os.path.exists(self.prf_config.rsq_map_mgh):
                nib.save(nib.freesurfer.mghformat.MGHImage(unmask_rsq.astype(np.float32, order = "C"),affine=affine),self.prf_config.rsq_map_mgh)




# class CfStimulus:

#     def __init__(self, dir_config, mri_config, prf_config, logger):
#         self.dir_config             = dir_config

#         self.screen_halfheight_cm   = prf_config.screen_halfheight_cm
#         self.screen_distance_cm     = prf_config.screen_distance_cm
#         self.TR                     = mri_config.TR

#         self.prf_run_config         = mri_config.prf_run_config
#         self.prfpy_output_config    = mri_config.prfpy_output_config
#         self.output_data_dict_fn    = prf_config.output_data_dict_fn

#         self.logger                 = logger
        
#         # Create stimulus object
#         self.logger.info('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
#         self.logger.info('Creating stimulus object')
#         self.logger.info('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
#         mri_config.prfpy_output_config  = self._create_cf_stim_obj()
#         self.logger.info('Stimulus object created')

#     def _create_cf_stim_obj(self):
#         """
#         Create a stimulus object for CF modeling.
#         """
#         from prfpy.stimulus import CFStimulus
