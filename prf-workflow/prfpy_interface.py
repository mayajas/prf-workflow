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
from scipy.optimize import NonlinearConstraint

def translate_indices_singlesurf(occ_mask, vert_centers_across_depth, depth, target_surfs):
    """
    Translate indices to single surface indices.
    """
    occ_vert_centres = vert_centers_across_depth
    if depth > 0:
        occ_vert_centres = [idx - len(occ_mask) * target_surfs.index(depth) for idx in occ_vert_centres]
        occ_vert_centres = np.array(occ_vert_centres)
        
    return occ_vert_centres

def vert_idx_constraint(params, vert_centers):
    vert_center = params[0]  # Extract the first parameter
    return 0 if vert_center in vert_centers else np.inf  # Return 0 if the parameter belongs to vert_centers, otherwise return infinity

class PrfpyStimulus:
    """
    Create a stimulus object for prfpy.
    Args:
        dir_config (DirConfig object): Directory configuration object.
        mri_config (MriConfig object): MRI configuration object.
        prf_config (PrfMappingConfig object): pRF mapping configuration object.
        logger (logging.Logger object): Logger object.
    """
    def __init__(self, dir_config, mri_config, prf_config, logger):
        self.dir_config             = dir_config
        self.screen_halfheight_cm   = prf_config.screen_halfheight_cm
        self.screen_distance_cm     = prf_config.screen_distance_cm
        self.TR                     = mri_config.TR

        self.prf_run_config         = mri_config.prf_run_config
        self.prfpy_output_config    = mri_config.prfpy_output_config
        self.output_data_dict_fn    = prf_config.output_data_dict_fn

        self.logger                 = logger
        
        # Create stimulus object
        self.logger.info('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        self.logger.info('Creating pRF stimulus object')
        self.logger.info('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        mri_config.prfpy_output_config  = self._create_stim_obj()
        self.logger.info('PRF stimulus object created')

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
        else:
            self.logger.info(f'Output data dict does not exist yet: {self.output_data_dict_fn}')
        
        # Define stimulus object for each aperture type (if 'combined' exists among keys, then only define 'combined' aperture)
        for aperture_type, config in self.prf_run_config.items():
            if 'combined' in self.prf_run_config.keys() and aperture_type != 'combined':
                continue
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
    """
    Fit pRF model.
    Args:
        dir_config (DirConfig object): Directory configuration object.
        mri_config (MriConfig object): MRI configuration object.
        prf_config (PrfMappingConfig object): pRF mapping configuration object.
        project_config (ProjectConfig object): Project configuration object.
        logger (logging.Logger object): Logger object.
    """
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

        # Input parameters of Iso2DGaussianModel
        self.hrf                        = prf_config.hrf
        self.filter_predictions         = prf_config.filter_predictions
        self.filter_type                = prf_config.filter_type
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
        self.pRF_param_pckl_fn          = prf_config.pRF_param_pckl_fn

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
                self._fit_dog(which_surf='avg')

                # then extract pRF parameters
                self.logger.info('Extracting pRF parameters for avg DoG model...')
                self._get_prf_params(which_model='DoG',which_surf='avg')

                # then fit DoG to each depth
                self.logger.info('%%%%%%%%% Fitting DoG model across depths %%%%%%%%%')
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
            if 'combined' in self.prf_run_config.keys() and aperture_type != 'combined':
                continue
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
                                                            n_jobs=self.n_procs, use_previous_gaussian_fitter_hrf=self.fit_hrf)


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
                                                                n_jobs=self.n_procs, use_previous_gaussian_fitter_hrf=self.fit_hrf)
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
                                                                n_jobs=self.n_procs, use_previous_gaussian_fitter_hrf=self.fit_hrf)
                        
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
            if 'combined' in self.prf_run_config.keys() and aperture_type != 'combined':
                continue
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
                                                        n_jobs=self.n_procs, use_previous_gaussian_fitter_hrf=self.fit_hrf,
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
                                                        n_jobs=self.n_procs, use_previous_gaussian_fitter_hrf=self.fit_hrf,
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


    def _get_prf_params(self,which_model,which_surf):
        """
        Extract pRF parameters from model fitter.
        """        
        
        # Extract pRF parameter estimates from iterative fit result
        if which_surf == 'single' or which_surf == 'avg':
            if which_model == 'Iso':
                pRF_param_fn = self.pRF_param_pckl_fn.format(which_surf='avg',which_model='')
            elif which_model == 'DoG':
                pRF_param_fn = self.pRF_param_pckl_fn.format(which_surf='avg',which_model='_DoG')

            # For single and average surfaces
            if not os.path.exists(pRF_param_fn):
                self.logger.info('{} does not yet exist'.format(pRF_param_fn))
                self.logger.info('Saving pRF parameters to pickle file...')
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
                
                for aperture_type in self.prf_run_config:
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
            if which_model == 'Iso':
                pRF_param_fn = self.pRF_param_pckl_fn.format(which_surf='per_depth',which_model='')
            elif which_model == 'DoG':
                pRF_param_fn = self.pRF_param_pckl_fn.format(which_surf='per_depth',which_model='_DoG')

            # For individual surfaces
            if not os.path.exists(pRF_param_fn):
                self.logger.info('{} does not yet exist'.format(pRF_param_fn))
                self.logger.info('Saving pRF parameters to pickle file...')

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

                for aperture_type in self.prf_run_config:
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
            occ_mask, n_vtx = pickle.load(pickle_file)

        if which_surf == 'single' or which_surf == 'avg':
            self.logger.info('Saving pRF parameters to mgh files for visualization...')

            for aperture_type in self.prf_run_config:
                self.logger.info(f"[[{aperture_type} aperture]]")
            
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
                

                unmask_x[occ_mask]              = prf_params[aperture_type]['x']
                unmask_y[occ_mask]              = prf_params[aperture_type]['y']
                unmask_prf_size[occ_mask]       = prf_params[aperture_type]['prf_size']
                unmask_prf_amp[occ_mask]        = prf_params[aperture_type]['prf_amp']
                unmask_bold_baseline[occ_mask]  = prf_params[aperture_type]['bold_baseline']
                unmask_srf_amp[occ_mask]        = prf_params[aperture_type]['srf_amp']
                unmask_srf_size[occ_mask]       = prf_params[aperture_type]['srf_size']
                unmask_hrf_1[occ_mask]          = prf_params[aperture_type]['hrf_1']
                unmask_hrf_2[occ_mask]          = prf_params[aperture_type]['hrf_2']
                unmask_rsq[occ_mask]            = prf_params[aperture_type]['total_rsq']
                unmask_polar[occ_mask]          = prf_params[aperture_type]['polar']
                unmask_ecc[occ_mask]            = prf_params[aperture_type]['ecc']

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
                unmask_x[np.isnan(unmask_x)]                            = 0.
                unmask_y[np.isnan(unmask_y)]                            = 0.
                unmask_prf_size[np.isnan(unmask_prf_size)]              = 0.
                unmask_prf_amp[np.isnan(unmask_prf_amp)]                = 0.
                unmask_bold_baseline[np.isnan(unmask_bold_baseline)]    = 0.
                unmask_srf_amp[np.isnan(unmask_srf_amp)]                = 0.
                unmask_srf_size[np.isnan(unmask_srf_size)]              = 0.
                unmask_hrf_1[np.isnan(unmask_hrf_1)]                    = 0.
                unmask_hrf_2[np.isnan(unmask_hrf_2)]                    = 0.
                unmask_rsq[np.isnan(unmask_rsq)]                        = 0.
                unmask_polar[np.isnan(unmask_polar)]                    = 0.
                unmask_ecc[np.isnan(unmask_ecc)]                        = 0.

                # Save maps to .mgh files for manual delineations
                meanFunc_mgh_nib = nib.freesurfer.mghformat.load(self.meanFunc_mgh_fn)
                affine = meanFunc_mgh_nib.affine

                # Prepare filenames
                depth = 'avg'
                x_map_mgh = self.prf_config.pRF_param_map_mgh.format(param_name='x',aperture_type=aperture_type,depth=depth)
                y_map_mgh = self.prf_config.pRF_param_map_mgh.format(param_name='y',aperture_type=aperture_type,depth=depth)
                prf_size_map_mgh = self.prf_config.pRF_param_map_mgh.format(param_name='prf_size',aperture_type=aperture_type,depth=depth)
                prf_amp_map_mgh = self.prf_config.pRF_param_map_mgh.format(param_name='prf_amp',aperture_type=aperture_type,depth=depth)
                bold_baseline_map_mgh = self.prf_config.pRF_param_map_mgh.format(param_name='bold_baseline',aperture_type=aperture_type,depth=depth)
                srf_amp_map_mgh = self.prf_config.pRF_param_map_mgh.format(param_name='srf_amp',aperture_type=aperture_type,depth=depth)
                srf_size_map_mgh = self.prf_config.pRF_param_map_mgh.format(param_name='srf_size',aperture_type=aperture_type,depth=depth)
                hrf_1_map_mgh = self.prf_config.pRF_param_map_mgh.format(param_name='hrf_1',aperture_type=aperture_type,depth=depth)
                hrf_2_map_mgh = self.prf_config.pRF_param_map_mgh.format(param_name='hrf_2',aperture_type=aperture_type,depth=depth)
                rsq_map_mgh = self.prf_config.pRF_param_map_mgh.format(param_name='rsq',aperture_type=aperture_type,depth=depth)
                polar_map_mgh = self.prf_config.pRF_param_map_mgh.format(param_name='pol',aperture_type=aperture_type,depth=depth)
                ecc_map_mgh = self.prf_config.pRF_param_map_mgh.format(param_name='ecc',aperture_type=aperture_type,depth=depth)

                # Save pRF parameters to mgh files for visualization
                if not os.path.exists(x_map_mgh) or self.prf_config.overwrite_viz:
                    nib.save(nib.freesurfer.mghformat.MGHImage(unmask_x.astype(np.float32, order = "C"),affine=affine),x_map_mgh)
                if not os.path.exists(y_map_mgh) or self.prf_config.overwrite_viz:
                    nib.save(nib.freesurfer.mghformat.MGHImage(unmask_y.astype(np.float32, order = "C"),affine=affine),y_map_mgh)
                if not os.path.exists(prf_size_map_mgh) or self.prf_config.overwrite_viz:
                    nib.save(nib.freesurfer.mghformat.MGHImage(unmask_prf_size.astype(np.float32, order = "C"),affine=affine),prf_size_map_mgh)
                if not os.path.exists(prf_amp_map_mgh) or self.prf_config.overwrite_viz:
                    nib.save(nib.freesurfer.mghformat.MGHImage(unmask_prf_amp.astype(np.float32, order = "C"),affine=affine),prf_amp_map_mgh)
                if not os.path.exists(bold_baseline_map_mgh) or self.prf_config.overwrite_viz:
                    nib.save(nib.freesurfer.mghformat.MGHImage(unmask_bold_baseline.astype(np.float32, order = "C"),affine=affine),bold_baseline_map_mgh)
                if not os.path.exists(srf_amp_map_mgh) or self.prf_config.overwrite_viz:
                    nib.save(nib.freesurfer.mghformat.MGHImage(unmask_srf_amp.astype(np.float32, order = "C"),affine=affine),srf_amp_map_mgh)
                if not os.path.exists(srf_size_map_mgh) or self.prf_config.overwrite_viz:
                    nib.save(nib.freesurfer.mghformat.MGHImage(unmask_srf_size.astype(np.float32, order = "C"),affine=affine),srf_size_map_mgh)
                if not os.path.exists(hrf_1_map_mgh) or self.prf_config.overwrite_viz:
                    nib.save(nib.freesurfer.mghformat.MGHImage(unmask_hrf_1.astype(np.float32, order = "C"),affine=affine),hrf_1_map_mgh)
                if not os.path.exists(hrf_2_map_mgh) or self.prf_config.overwrite_viz:
                    nib.save(nib.freesurfer.mghformat.MGHImage(unmask_hrf_2.astype(np.float32, order = "C"),affine=affine),hrf_2_map_mgh)
                if not os.path.exists(polar_map_mgh) or self.prf_config.overwrite_viz:
                    nib.save(nib.freesurfer.mghformat.MGHImage(unmask_polar.astype(np.float32, order = "C"),affine=affine),polar_map_mgh)
                if not os.path.exists(ecc_map_mgh) or self.prf_config.overwrite_viz:
                    nib.save(nib.freesurfer.mghformat.MGHImage(unmask_ecc.astype(np.float32, order = "C"),affine=affine),ecc_map_mgh)
                if not os.path.exists(rsq_map_mgh) or self.prf_config.overwrite_viz:
                    nib.save(nib.freesurfer.mghformat.MGHImage(unmask_rsq.astype(np.float32, order = "C"),affine=affine),rsq_map_mgh)

class CfStimulus:
    """
    Create stimulus object for CF modeling.
    Args:
        mri_config (MriConfig object): MRI configuration object.
        cfm_config (CfmConfig object): CF modeling configuration object.
        logger (Logger object): Logger object.
    """

    def __init__(self, mri_config, cfm_config, logger):
        self.cf_run_config          = mri_config.cf_run_config
        self.cfm_output_config      = mri_config.cfm_output_config
        self.occ_mask_fn            = mri_config.occ_mask_fn
        self.output_data_dict_fn    = cfm_config.output_data_dict_fn

        self.logger             = logger
        
        # Create stimulus object
        self.logger.info('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        self.logger.info('Creating CF stimulus object')
        self.logger.info('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        mri_config.cfm_output_config  = self._create_cf_stim_obj()
        self.logger.info('CF stimulus object created')

    def _create_cf_stim_obj(self):
        """
        Create a stimulus object for CF modeling.
        """
        from prfpy.stimulus import CFStimulus

        # First, try loading the output data dict
        if os.path.exists(self.output_data_dict_fn):
            self.logger.info(f'Output data dict already exists: {self.output_data_dict_fn}')
            self.logger.info('Loading output data dict...')
            with open(self.output_data_dict_fn, 'rb') as pickle_file:
                self.cfm_output_config = pickle.load(pickle_file)
            self.logger.info('Loaded.')
        else:
            self.logger.error(f'Output data dict does not exist: {self.output_data_dict_fn}')
            sys.exit(1)
        
        # Define stimulus object for each aperture type and each subsurface
        for aperture_type, config in self.cfm_output_config.items():
            self.logger.info('Creating CFM stimulus for aperture type: {}'.format(aperture_type))
            for subsurf_name, subsurface in config.items():
                self.logger.info('Subsurface: {}'.format(subsurf_name))

                if not subsurface['stim']:
                    self.cfm_output_config[aperture_type][subsurf_name]['stim'] = CFStimulus(subsurface['data'],subsurface['subsurface_translated'],subsurface['dist'])

                    ## Save subsurfaces
                    self.logger.info('Saving subsurface stimulus...')
                    with open(self.output_data_dict_fn, 'wb') as pickle_file:
                        pickle.dump(self.cfm_output_config, pickle_file)
                else:
                    self.logger.info('Stimulus object already defined')
            
        return self.cfm_output_config
    
class CfModeling:
    """
    Do connective field model fitting
    Args:
        project_config (ProjectConfig object): Project configuration object.
        mri_config (MriConfig object): MRI configuration object.
        cfm_config (CfmConfig object): CF modeling configuration object.
        logger (Logger object): Logger object.
    """

    def __init__(self, project_config, mri_config, cfm_config, logger):
        # Load configuration parameters
        self.cf_run_config          = mri_config.cf_run_config
        self.cfm_output_config      = mri_config.cfm_output_config
        self.occ_mask_fn            = mri_config.occ_mask_fn
        self.cfm_config             = cfm_config
        self.target_surfs           = cfm_config.target_surfs
        self.output_data_dict_fn    = cfm_config.output_data_dict_fn
        self.sigmas                 = cfm_config.CF_sizes
        self.verbose                = cfm_config.verbose
        self.rsq_thresh_itfit       = cfm_config.rsq_thresh_itfit
        self.use_bounds             = cfm_config.use_bounds
        self.use_constraints        = cfm_config.use_constraints

        # Number of cores to use for parallel processing of vertices, number of surfaces
        self.n_procs                = project_config.n_procs
        self.n_surfs                = project_config.n_surfs

        # Logger
        self.logger                 = logger

        # CFM output filenames
        self.cfm_param_fn           = cfm_config.cfm_param_fn

        ## Load occipital mask
        logger.info('Loading occipital mask...')
        with open(self.occ_mask_fn, 'rb') as pickle_file:
            [self.occ_mask,_] = pickle.load(pickle_file)
        
        # Fit CF model
        self.logger.info('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        self.logger.info('Fitting CF model')
        self.logger.info('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        self.logger.info('Nr processors: ' + str(self.n_procs))
        self._cfm_fitting()
        self.logger.info('CF model fit complete')

    def _cfm_fitting(self):
        """
        Main CF model fitting function.
        """
        # Fit the CF model
        self.logger.info('%%%%%%%%% Fitting CF model %%%%%%%%%')
        self._fit_cf_model()

        # then extract CFM parameters
        self.logger.info('Extracting CFM parameters...')
        self._get_cfm_params()
    
    def _define_bounds_and_constraints(self,aperture_type,subsurf_name):
        """
        Define bounds for CF model fitting.
        Parameter order: 'vert_centers','prf_size','beta','baseline','total_rsq'
        """
        vert_centers = np.array(self.cfm_output_config[aperture_type][subsurf_name]['subsurface_translated'])
        vtx_centers_gridfit = self.cfm_output_config[aperture_type][subsurf_name]['gf'].gridsearch_params[:,0]
        n_params = self.cfm_output_config[aperture_type][subsurf_name]['gf'].gridsearch_params.shape[1]-1 # last param is rsq (exclude)

        if self.use_bounds:
            bounds = [
                (vtx_centers_gridfit[i], vtx_centers_gridfit[i]) if i == 0 else (None, None)
                for i in range(n_params)
            ] 
        else:
            bounds = None

        if self.use_constraints:
            #nonlinear_constraint_obj = NonlinearConstraint(lambda params: vert_idx_constraint(params, vert_centers), 0, 0)
            
            # Update the constraints list with the dictionary representation of the constraint
            # constraints = [NonlinearConstraint(lambda params: vert_idx_constraint(params, vert_centers), 
            #                                    lb=min_vtx, ub=max_vtx, keep_feasible=True)]
            constraints = {'type': 'eq', 'fun': lambda params: vert_idx_constraint(params, vert_centers)}
        else:
            constraints = None

        return bounds, constraints

    def _fit_cf_model(self):
        """
        Fit CF model.
        """
        from prfpy.model import CFGaussianModel
        from prfpy.fit import CFFitter

        # First, try loading the output data dict
        if os.path.exists(self.output_data_dict_fn):
            self.logger.info(f'Output data dict already exists: {self.output_data_dict_fn}')
            self.logger.info('Loading output data dict...')
            with open(self.output_data_dict_fn, 'rb') as pickle_file:
                self.cfm_output_config = pickle.load(pickle_file)
            self.logger.info('Loaded.')
        else:
            self.logger.error(f'Output data dict does not exist: {self.output_data_dict_fn}')
            sys.exit(1)

        # Define CF model
        for aperture_type, config in self.cfm_output_config.items():
            self.logger.info('Defining CF model for aperture type: {}'.format(aperture_type))
            for subsurf_name, subsurface in config.items():
                self.logger.info('Subsurface: {}'.format(subsurf_name))

                if not subsurface['model']:
                    self.cfm_output_config[aperture_type][subsurf_name]['model'] = CFGaussianModel(subsurface['stim'])

                    ## Save subsurfaces
                    self.logger.info('Saving subsurface CF model...')
                    with open(self.output_data_dict_fn, 'wb') as pickle_file:
                        pickle.dump(self.cfm_output_config, pickle_file)
                else:
                    self.logger.info('CF model already defined')

        # Define CF model fitter
        for aperture_type, config in self.cfm_output_config.items():
            self.logger.info('Defining CF model fitter for aperture type: {}'.format(aperture_type))
            for subsurf_name, subsurface in config.items():
                self.logger.info('Subsurface: {}'.format(subsurf_name))

                if not subsurface['gf']:
                    self.cfm_output_config[aperture_type][subsurf_name]['gf'] = CFFitter(data=self.cf_run_config[aperture_type]['preproc_data_concatenated_depths'],
                                                                                         model=subsurface['model'],
                                                                                         fit_hrf=False) #TODO: add HRF fitting

                    ## Save subsurfaces
                    self.logger.info('Saving subsurface CF model...')
                    with open(self.output_data_dict_fn, 'wb') as pickle_file:
                        pickle.dump(self.cfm_output_config, pickle_file)
                else:
                    self.logger.info('CF model fitter already defined')

        # Fit CF model: grid fit
        for aperture_type, config in self.cfm_output_config.items():
            self.logger.info('Grid fitting CF model for aperture type: {}'.format(aperture_type))
            for subsurf_name, subsurface in config.items():
                self.logger.info('Subsurface: {}'.format(subsurf_name))

                if not subsurface['is_gf']['gridfit']:
                    self.cfm_output_config[aperture_type][subsurf_name]['gf'].grid_fit(sigma_grid = self.sigmas, 
                                                                                       verbose=self.verbose, 
                                                                                       n_batches=self.n_procs)
                    self.logger.info('CF model grid fit complete.')
                    self.cfm_output_config[aperture_type][subsurf_name]['is_gf']['gridfit'] = True

                    ## Save subsurfaces
                    self.logger.info('Saving subsurface CF model...')
                    with open(self.output_data_dict_fn, 'wb') as pickle_file:
                        pickle.dump(self.cfm_output_config, pickle_file)
                else:
                    self.logger.info('CF model grid fit already completed')

        # Fit CF model: iterative fit
        for aperture_type, config in self.cfm_output_config.items():
            self.logger.info('Iterative fitting CF model for aperture type: {}'.format(aperture_type))
            for subsurf_name, subsurface in config.items():
                self.logger.info('Subsurface: {}'.format(subsurf_name))

                if not subsurface['is_gf']['itfit']:
                    # Define bounds and constraints, if any
                    bounds, constraints = self._define_bounds_and_constraints(aperture_type,subsurf_name)
                    
                    # Fit CF model: iterative fit
                    self.cfm_output_config[aperture_type][subsurf_name]['gf'].iterative_fit(rsq_threshold=self.rsq_thresh_itfit, 
                                                                                            verbose=self.verbose,
                                                                                            bounds=bounds,
                                                                                            constraints=constraints)
                    self.logger.info('CF model iterative fit complete.')
                    self.cfm_output_config[aperture_type][subsurf_name]['is_gf']['itfit'] = True

                    ## Save subsurfaces
                    self.logger.info('Saving subsurface CF model...')
                    with open(self.output_data_dict_fn, 'wb') as pickle_file:
                        pickle.dump(self.cfm_output_config, pickle_file)
                else:
                    self.logger.info('CF model iterative fit already completed')

    def _get_cfm_params(self):
        """
        Extract CFM parameters from model fitter.
        """

        # Extract pRF parameter estimates from iterative fit result
        if not os.path.exists(self.cfm_param_fn):
            self.logger.info('{} does not yet exist'.format(self.cfm_param_fn))
            # For single target surfaces
            if len(self.target_surfs) == 1:
                # Initialize pRF parameters 
                cf_params      = {
                    key:{
                        subsurf: {
                            'vert_centers': [],
                            'prf_size': [],
                            'beta': [],
                            'baseline': [],
                            'total_rsq': []
                        } for subsurf in self.cfm_config.subsurfaces
                    } for key in self.cf_run_config
                }

                for aperture_type, config in self.cfm_output_config.items():
                    self.logger.info(f"[[{aperture_type} aperture]]")
                    for subsurf_name, subsurface in config.items():
                        self.logger.info(f"[[{subsurf_name} subsurface]]")

                        # Extract iterative search parameters
                        vert_centers    = subsurface['gf'].iterative_search_params[:,0]
                        prf_size        = subsurface['gf'].iterative_search_params[:,1]
                        beta            = subsurface['gf'].iterative_search_params[:,2]
                        baseline        = subsurface['gf'].iterative_search_params[:,3]
                        total_rsq       = subsurface['gf'].iterative_search_params[:,-1]

                        # Translate vert_centers back to single surface indices
                        vert_centers = translate_indices_singlesurf(occ_mask = self.occ_mask, vert_centers_across_depth = vert_centers, 
                                               depth = subsurface['depth'], target_surfs = self.target_surfs)

                        # Save parameters
                        cf_params[aperture_type][subsurf_name]['vert_centers']   = vert_centers
                        cf_params[aperture_type][subsurf_name]['prf_size']       = prf_size
                        cf_params[aperture_type][subsurf_name]['beta']           = beta
                        cf_params[aperture_type][subsurf_name]['baseline']       = baseline
                        cf_params[aperture_type][subsurf_name]['total_rsq']      = total_rsq

            # For multiple target surfaces
            elif len(self.target_surfs) > 1:
                # Initialize pRF parameters 
                cf_params      = {
                    key:{
                        subsurf: {
                            'vert_centers': [0] * self.n_surfs,
                            'prf_size': [0] * self.n_surfs,
                            'beta': [0] * self.n_surfs,
                            'baseline': [0] * self.n_surfs,
                            'total_rsq': [0] * self.n_surfs
                        } for subsurf in self.cfm_config.subsurfaces
                    } for key in self.cf_run_config
                }

                for aperture_type, config in self.cfm_output_config.items():
                    self.logger.info(f"[[{aperture_type} aperture]]")
                    for subsurf_name, subsurface in config.items():
                        self.logger.info(f"[[{subsurf_name} subsurface]]")

                        # Extract iterative search parameters
                        vert_centers    = subsurface['gf'].iterative_search_params[:,0]
                        prf_size        = subsurface['gf'].iterative_search_params[:,1]
                        beta            = subsurface['gf'].iterative_search_params[:,2]
                        baseline        = subsurface['gf'].iterative_search_params[:,3]
                        total_rsq       = subsurface['gf'].iterative_search_params[:,-1]

                        # Translate vert_centers back to single surface indices
                        vert_centers = translate_indices_singlesurf(occ_mask = self.occ_mask, vert_centers_across_depth = vert_centers, 
                                               depth = subsurface['depth'], target_surfs = self.target_surfs)

                        # Reshape vert_centers, prf_size, beta, baseline, total_rsq to n_surfs by n_vtx
                        vert_centers = vert_centers.reshape(self.n_surfs,-1)
                        prf_size = prf_size.reshape(self.n_surfs,-1)
                        beta = beta.reshape(self.n_surfs,-1)
                        baseline = baseline.reshape(self.n_surfs,-1)
                        total_rsq = total_rsq.reshape(self.n_surfs,-1)  

                        # Save parameters
                        for depth in range(0,self.n_surfs):
                            cf_params[aperture_type][subsurf_name]['vert_centers'][depth]   = vert_centers[depth,:]
                            cf_params[aperture_type][subsurf_name]['prf_size'][depth]       = prf_size[depth,:]
                            cf_params[aperture_type][subsurf_name]['beta'][depth]           = beta[depth,:]
                            cf_params[aperture_type][subsurf_name]['baseline'][depth]       = baseline[depth,:]
                            cf_params[aperture_type][subsurf_name]['total_rsq'][depth]      = total_rsq[depth,:]

            # Save pRF parameters
            with open(self.cfm_param_fn, 'wb') as pickle_file:
                pickle.dump(cf_params, pickle_file)        