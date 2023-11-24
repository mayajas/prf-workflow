# config.py

import json
import math
import numpy as np
import os
from os.path import join as opj
import scipy
import sys


def replace_placeholders(config, replacements):
    if isinstance(config, dict):
        for key, value in config.items():
            config[key] = replace_placeholders(value, replacements)
    elif isinstance(config, str):
        for placeholder, replacement in replacements.items():
            config = config.replace("{" + placeholder + "}", str(replacement))
    return config

class ProjectConfig:
    """
    This class contains project-specific information.

    Attributes:
        proj_id (str): name of the project
        subject_list (list): list of all subjects in the project
        hem_list (list): list of all hemispheres to be analyzed in the project
    """
    
    def __init__(self, config_file, sub_idx, hem_idx):
        # get config from config file
        self.subject_list, self.hem_list, self.n_surfs, self.logger_dir = self._load_config(config_file)
        
        self.subject_id = self.subject_list[sub_idx]
        self.hemi = self.hem_list[hem_idx]

        # Determine the number of cores
        try:
            self.n_procs = int(os.getenv('OMP_NUM_THREADS'))
        except TypeError:
            self.n_procs = 1 

    def _load_config(self, config_file):
        with open(config_file) as f:
            config_data = json.load(f)
        
        class_name = self.__class__.__name__
        class_section = config_data.get(class_name, {})

        self.subject_list = class_section.get('subject_list', ['sub-01', 'sub-02', 'sub-03'])
        self.hem_list = class_section.get('hem_list', ['lh', 'rh'])
        self.n_surfs = class_section.get('n_surfs', 1)
        self.logger_dir = class_section.get('logger_dir', None)

        return self.subject_list, self.hem_list, self.n_surfs, self.logger_dir


class DirConfig:
    def __init__(self, config_file, project_config, logger):
        self.subject_id     = project_config.subject_id
        self.logger         = logger

        # get config from config file
        self.FS_dir, self.prf_output_dir, self.apertures_dir, self.surface_tools_dir = self._load_config(config_file)  

    def _load_config(self, config_file):
        with open(config_file) as f:
            config_data = json.load(f)
        
        class_name = self.__class__.__name__
        class_section = config_data.get(class_name, {})

        # Replace placeholders in the configuration data
        replacements = {"subject_id": self.subject_id}
        class_section = replace_placeholders(class_section, replacements)

        # freesurfer directory
        self.FS_dir = class_section.get('FS_dir', None)
        
        # output directory
        self.prf_output_dir = class_section.get('prf_output_dir', None)
        
        # path to stimulus apertures mat files
        self.apertures_dir = class_section.get('apertures_dir', None)
        
        # path to surface tools 
        self.surface_tools_dir = class_section.get('surface_tools_dir', None)

        # check if directories exist, write errors to logger, and return directories
        if not os.path.exists(self.FS_dir):
            self.logger.error('Freesurfer directory does not exist.')
            sys.exit(1)
        else:
            self.logger.info('Freesurfer directory: ' + self.FS_dir)
        if not os.path.exists(self.prf_output_dir):
            self.logger.error('Output directory does not exist.')
            sys.exit(1)
        else: 
            self.logger.info('Output directory: ' + self.prf_output_dir)
        if not os.path.exists(self.apertures_dir):
            self.logger.error('Apertures directory does not exist.')
            sys.exit(1)
        else:
            self.logger.info('Apertures directory: ' + self.apertures_dir)
        if not os.path.exists(self.surface_tools_dir):
            self.logger.error('Surface tools directory does not exist.')
            sys.exit(1)
        else:
            self.logger.info('Surface tools directory: ' + self.surface_tools_dir)

        return self.FS_dir, self.prf_output_dir, self.apertures_dir, self.surface_tools_dir

class PrfMappingConfig:
    """
    This class contains pRF mapping-related information.
    """
    def __init__(self, config_file, dir_config, project_config, logger):
        self.prf_output_dir   = dir_config.prf_output_dir
        self.subject_id = project_config.subject_id
        self.hemi       = project_config.hemi
        self.n_surfs    = project_config.n_surfs
        self.logger     = logger
        
        # get config from config file
        self.screen_height_cm, self.screen_distance_cm, self.which_model, self.avg_runs, self.fit_hrf, self.start_from_avg, self.fit_css, self.grid_nr, self.y_coord_cutoff, self.verbose, self.hrf, self.filter_predictions, self.filter_type, self.filter_params, self.normalize_RFs, self.rsq_thresh_itfit, self.rsq_thresh_viz, self.reference_aperture = \
            self._load_config(config_file)

        # calculate screen dimensions
        self._get_screen_dimensions()

        # get grid search parameters
        self.size_grid, self.ecc_grid, self.polar_grid, self.surround_amplitude_grid, self.surround_size_grid = self._get_grid_search_params()

        # initialize pRF model preferences and output directory name
        self.model_name, self.avg_runs, self.out_dir = self._get_prf_mapping_config()
        
        # initialize prf output filenames
        self.input_data_dict_fn, self.output_data_dict_fn, self.pRF_param_avg_fn, self.x_map_mgh, self.y_map_mgh, self.prf_size_map_mgh, self.prf_amp_map_mgh, self.bold_baseline_map_mgh, self.srf_amp_map_mgh, self.srf_size_map_mgh, self.hrf_1_map_mgh, self.hrf_2_map_mgh, self.rsq_map_mgh, self.polar_map_mgh, self.ecc_map_mgh, self.pRF_param_per_depth_fn, self.polar_map_per_depth_mgh, self.ecc_map_per_depth_mgh, self.hrf_1_map_per_depth_mgh, self.hrf_2_map_per_depth_mgh = self._get_prf_output_fns()
         
    def _load_config(self, config_file):
        with open(config_file) as f:
            config_data = json.load(f)
        
        class_name = self.__class__.__name__
        class_section = config_data.get(class_name, {})

        # full screen height (cm)
        self.screen_height_cm = class_section.get('screen_height_cm', 12.00)

        # distance from screen (cm)
        self.screen_distance_cm = class_section.get('screen_distance_cm', 12.00)

        # pRF model preferences
        self.which_model    = class_section.get('which_model', 'Iso')   # 'Iso' or 'DoG'
        self.avg_runs       = class_section.get('avg_runs', True)       # boolean
                                                                        # whether or not to average runs of the same aperture type
                                                                        # if False, then runs are concatenated TODO: implement this option
        self.fit_hrf        = class_section.get('fit_hrf', False)       # boolean
                                                                        # whether or not to fit two extra parameters for hrf derivative and
                                                                        # dispersion.
        self.start_from_avg = class_section.get('start_from_avg', True) # boolean
                                                                        # whether to use avg across depths as starting point for layer fits
                                                                        # when not fitting layer-specific prfs, this parameter is meaningless
                                                                        # as prf runs are projected to the gm surface and fitting is done there (only once)
        self.fit_css        = class_section.get('fit_css', False)       # boolean
                                                                        # whether or not to fit css model


        # size of grid for pRF mapping initial grid search
        self.grid_nr        = class_section.get('grid_nr', 30)
        
        # occipital mask y-coordinate cut-off (including only posterior vertices)
        self.y_coord_cutoff = class_section.get('y_coord_cutoff', -25)  # careful when choosing this - use inflated surface on CURTA in jupyter notebook to check

        # Input parameters of Iso2DGaussianModel
        self.verbose = class_section.get('verbose', True)               # boolean, optional
                                                                        # whether to print out progress messages            
        self.hrf = class_section.get('hrf', None)                       # string, list or numpy.ndarray, optional
                                                                        # HRF shape for this Model.
                                                                        # Can be 'direct', which implements nothing (for eCoG or later convolution),
                                                                        # a list or array of 3, which are multiplied with the three spm HRF basis functions,
                                                                        # and an array already sampled on the TR by the user.
                                                                        # (the default is None, which implements standard spm HRF)
        self.filter_predictions = class_section.get('filter_predictions', False) # boolean, optional
                                                                                # whether to high-pass filter the predictions, default False        
        self.filter_type = class_section.get('filter_type', 'sg')         # string
                                                                        # type of filter to use for high-pass filtering
                                                                        # options: 'sg' (Savitzky-Golay) or 'butterworth'   
        sg_filter_window_length = 201
        sg_filter_polyorder     = 3        
        self.filter_params = class_section.get('filter_params', {'window_length':sg_filter_window_length, 
                                                                 'polyorder':sg_filter_polyorder}) # dict
        self.normalize_RFs = class_section.get('normalize_RFs', False)   # boolean, optional
                                                                        # whether to normalize the RF volumes (generally not needed).                               

        # Input parameters for iterative fit
        self.rsq_thresh_itfit = class_section.get('rsq_thresh_itfit', 0.0005) # float
                                                                                # Rsq threshold for iterative fitting. Must be between 0 and 1.         
        self.rsq_thresh_viz = class_section.get('rsq_thresh_viz', 0.2)         # float
                                                                                # Rsq threshold for visualization. Must be between 0 and 1.   
        self.reference_aperture = class_section.get('reference_aperture',  None) # if not None, the pRF model fit from this aperture will be used to initialize
                                                                                    # the fitting for other apertures
                                                                                    # TODO: make sure reference aperture is first in prf_run_config so that it is estimated first
        self.logger.info('Selected reference aperture: '+self.reference_aperture)
                                                                                                                                    

        return self.screen_height_cm, self.screen_distance_cm, self.which_model, self.avg_runs, self.fit_hrf, self.start_from_avg, self.fit_css, self.grid_nr, self.y_coord_cutoff, self.verbose, self.hrf, self.filter_predictions, self.filter_type, self.filter_params, self.normalize_RFs, self.rsq_thresh_itfit, self.rsq_thresh_viz, self.reference_aperture


    def _get_screen_dimensions(self):
        # pRF mapping stimulus dimensions
        self.screen_halfheight_cm = self.screen_height_cm/2
        self.max_ecc                 = math.atan(self.screen_halfheight_cm/self.screen_distance_cm)
        self.max_ecc_deg             = math.degrees(self.max_ecc)
        self.max_ecc_size            = round(self.max_ecc_deg,2)
    
   
    def _get_grid_search_params(self):
        # grid search parameters
        size_grid, ecc_grid, polar_grid = self.max_ecc_size * np.linspace(0.25,1,self.grid_nr)**2, \
                            self.max_ecc_size * np.linspace(0.1,1,self.grid_nr)**2, \
                                np.linspace(0, 2*np.pi, self.grid_nr)
        
        if self.which_model == 'DoG':
            surround_amplitude_grid, surround_size_grid = np.linspace(0.25,1,self.grid_nr)**2, \
                    3*self.max_ecc_size * np.linspace(0.25,1,self.grid_nr)**2
        else:
            surround_amplitude_grid, surround_size_grid = None, None
        
        return size_grid, ecc_grid, polar_grid, surround_amplitude_grid, surround_size_grid
        

    # prf mapping preferences
    def _get_prf_mapping_config(self):

        # model name (based on above preferences)
        if self.n_surfs > 1:
            model_name     = 'prf_'+self.which_model+'_fit_hrf_'+str(self.fit_hrf)+'_start_from_avg_'+str(self.start_from_avg)+'_n_surfs_'+str(self.n_surfs)
            out_dir        = opj(self.prf_output_dir,model_name,self.subject_id)
        else:
            self.start_from_avg = None
            model_name          = 'prf_'+self.which_model+'_fit_hrf_'+str(self.fit_hrf)
            out_dir             = opj(self.prf_output_dir,model_name,self.subject_id)
        
        # check if out_dir exists, if not, create it
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        return model_name, self.avg_runs, out_dir
        
    def _get_prf_output_fns(self):
        # data dictionary files
        input_data_dict_fn      = opj(self.out_dir,self.hemi+'_input_data.pckl')
        output_data_dict_fn     = opj(self.out_dir,self.hemi+'_output_data.pckl')

        # avg across depths
        # prfpy outputs
        pRF_param_avg_fn        = opj(self.out_dir,self.hemi+'_pRF_params_avg')

        x_map_mgh               = opj(self.out_dir,self.hemi+'.x.mgh')
        y_map_mgh               = opj(self.out_dir,self.hemi+'.y.mgh')
        prf_size_map_mgh        = opj(self.out_dir,self.hemi+'.prf_size.mgh')
        prf_amp_map_mgh         = opj(self.out_dir,self.hemi+'.prf_amp.mgh')
        bold_baseline_map_mgh   = opj(self.out_dir,self.hemi+'.bold_baseline.mgh')
        srf_amp_map_mgh         = opj(self.out_dir,self.hemi+'.srf_amp.mgh')
        srf_size_map_mgh        = opj(self.out_dir,self.hemi+'.srf_size.mgh')
        hrf_1_map_mgh           = opj(self.out_dir,self.hemi+'.hrf_1.mgh')
        hrf_2_map_mgh           = opj(self.out_dir,self.hemi+'.hrf_2.mgh')
        polar_map_mgh           = opj(self.out_dir,self.hemi+'.pol.mgh')
        ecc_map_mgh             = opj(self.out_dir,self.hemi+'.ecc.mgh')
        rsq_map_mgh             = opj(self.out_dir,self.hemi+'.rsq.mgh')
        
        # layer-specific
        if self.n_surfs > 1:
            # prfpy outputs
            pRF_param_per_depth_fn      = opj(self.out_dir,self.hemi+'_pRF_params_per_depth')

            # prf parameter maps
            polar_map_per_depth_mgh     = [opj(self.out_dir,self.hemi+'.pol.'+str(depth)+'.mgh')
                                            for depth in range(0,self.n_surfs)]
            ecc_map_per_depth_mgh       = [opj(self.out_dir,self.hemi+'.ecc.'+str(depth)+'.mgh')
                                            for depth in range(0,self.n_surfs)]
            if self.fit_hrf:
                hrf_1_map_per_depth_mgh = [opj(self.out_dir,self.hemi+'.hrf_1.'+str(depth)+'.mgh')
                                            for depth in range(0,self.n_surfs)]
                hrf_2_map_per_depth_mgh = [opj(self.out_dir,self.hemi+'.hrf_2.'+str(depth)+'.mgh')
                                            for depth in range(0,self.n_surfs)]
            else:
                hrf_1_map_per_depth_mgh = None
                hrf_2_map_per_depth_mgh     = None
        else:
            pRF_param_per_depth_fn      = None
            polar_map_per_depth_mgh     = None
            ecc_map_per_depth_mgh       = None
            hrf_1_map_per_depth_mgh     = None
            hrf_2_map_per_depth_mgh     = None

        return input_data_dict_fn, output_data_dict_fn, pRF_param_avg_fn, x_map_mgh, y_map_mgh, prf_size_map_mgh, prf_amp_map_mgh, bold_baseline_map_mgh, srf_amp_map_mgh, srf_size_map_mgh, hrf_1_map_mgh, hrf_2_map_mgh, rsq_map_mgh, polar_map_mgh, ecc_map_mgh, pRF_param_per_depth_fn, polar_map_per_depth_mgh, ecc_map_per_depth_mgh, hrf_1_map_per_depth_mgh, hrf_2_map_per_depth_mgh

class MriConfig:
    """
    This class contains MRI acquisition-related information.
    """

    def __init__(self, config_file, project_config, dir_config, prf_config, logger):
        self.prf_output_dir     = dir_config.prf_output_dir
        self.reference_aperture = prf_config.reference_aperture
        self.FS_dir             = dir_config.FS_dir
        self.subject_id         = project_config.subject_id
        self.hemi               = project_config.hemi
        self.n_surfs            = project_config.n_surfs
        self.logger             = logger

        # get config from config file
        self.TR, self.equivol_fn, self.meanFunc_nii_fn, self.prf_run_config = self._load_config(config_file)

        # get input mri filenames
        self.gm_surf_fn, self.wm_surf_fn, self.inflated_surf_fn, self.equi_surf_fn_list, self.meanFunc_mgh_fn, self.occ_mask_fn = self._get_mri_fns()

        # get info about pRF runs (apertures, nr sessions per run, filenames of projected runs)
        self.prf_run_config, self.prfpy_output_config = self._get_prf_run_list()


    def _load_config(self, config_file):
        with open(config_file) as f:
            config_data = json.load(f)

        class_name = self.__class__.__name__
        class_section = config_data.get(class_name, {})

        # Replace placeholders in the configuration data
        replacements = {"subject_id": self.subject_id}
        class_section = replace_placeholders(class_section, replacements)

        self.TR = class_section.get('TR', 2.0)
        self.equivol_fn = class_section.get('equivol_fn', 'equi') # equivolumetric surface filename prefix
        self.meanFunc_nii_fn = class_section.get('meanFunc_nii_fn', None) # mean functional nitfti filepath and name
        self.prf_run_config  = class_section.get('prf_run_config',  None) # dictionary containing info about pRF runs
        
        # Check that mean functional and pRF nifti files exist
        if not os.path.exists(self.meanFunc_nii_fn):
            self.logger.error('Mean functional image does not exist under this address: '+ self.meanFunc_nii_fn)
        else:
            self.logger.info('Mean functional image: ' + self.meanFunc_nii_fn)
            sys.exit(1)
        for aperture_type, config in self.prf_run_config.items():
            for run in range(config['n_runs']):
                if not os.path.exists(config['nii_fn_list'][run]):
                    self.logger.error('PRF '+aperture_type+' run '+str(run)+' does not exist under this address: '+config['nii_fn_list'][run])
                    sys.exit(1)
                else:
                    self.logger.info('PRF '+aperture_type+' run '+str(run)+': '+config['nii_fn_list'][run])

        # Check that the reference aperture is among the apertures provided in prf_run_config
        if (self.reference_aperture is not None) and (self.reference_aperture not in self.prf_run_config.items()):
            self.logger.error('The selected reference aperture ('+self.reference_aperture+') is not present in the list of all stimulus apertures.')
            sys.exit(1)

        return self.TR, self.equivol_fn, self.meanFunc_nii_fn, self.prf_run_config
    
    def _get_mri_fns(self):
        # Freesurfer mesh filenames
        gm_surf_fn          = opj(self.FS_dir,self.subject_id,'surf',self.hemi+'.pial')
        wm_surf_fn          = opj(self.FS_dir,self.subject_id,'surf',self.hemi+'.white')
        inflated_surf_fn    = opj(self.FS_dir,self.subject_id,'surf',self.hemi+'.inflated')

        # equivolumetric surface output filenames
        if self.n_surfs > 1:
            equi_surf_fn_list = []
            for depth in range(self.n_surfs):         
                equi_surf_fn_list.append(self.equivol_fn+'{}.pial'.format(str(float(depth)/(self.n_surfs-1))))
            equi_surf_fn_list.reverse() # reverse order of surfaces so that pial is first
        else:
            equi_surf_fn_list = None

        # surface-projected functional run filenames
        meanFunc_mgh_fn  = opj(self.prf_output_dir,self.hemi+'.meanFunc.mgh')

        # occipital mask filename
        occ_mask_fn             = opj(self.prf_output_dir,self.hemi+'_occ_mask.pckl')

        return gm_surf_fn, wm_surf_fn, inflated_surf_fn, equi_surf_fn_list, meanFunc_mgh_fn, occ_mask_fn
    
    def _get_prf_run_list(self):
        for aperture_type, config in self.prf_run_config.items():
            mgh_fn_list = []
            for run in range(config['n_runs']):
                mgh_fn = []
                for depth in range(self.n_surfs if self.n_surfs > 1 else 1):
                    mgh_fn.append(opj(self.prf_output_dir, f"{self.hemi}.equi{depth}.{aperture_type}{run + 1}.mgh"))
                mgh_fn_list.append(mgh_fn)

            config['mgh_fn_list'] = mgh_fn_list

        prfpy_output_config = {
            key: {
                'stim': [],
                'is_gg': {
                    'avg': False,
                    'dog_avg': False
                },
                'is_gf': {
                    'avg': {
                        'gridfit': False,
                        'itfit': False
                    },
                    'per_depth': {
                        'gridfit': [False] * self.n_surfs,
                        'itfit': [False] * self.n_surfs
                    },
                    'dog_avg': {
                        'gridfit': False,
                        'itfit': False
                    },
                    'dog_per_depth': {
                        'gridfit': [False] * self.n_surfs,
                        'itfit': [False] * self.n_surfs
                    }
                },
                'gg_avg': [],
                'gf_avg': [],
                'gf_per_depth': [0] * self.n_surfs,
                'gg_dog_avg': [],
                'gf_dog_avg': [],
                'gf_dog_per_depth': [0] * self.n_surfs
            } for key in self.prf_run_config
        }

        return self.prf_run_config, prfpy_output_config

class StimApertureConfig:
    """
    This class loads the stimulus apertures into the prf_run_config dictionary.
    """

    def __init__(self, dir_config, mri_config, logger):
        self.logger         = logger
        self.prf_run_config = mri_config.prf_run_config
        self.apertures_dir  = dir_config.apertures_dir
        
        # get design matricies
        self._get_design_mats()

    def _get_design_mats(self):
        for stimulus_type, stimulus_config in self.prf_run_config.items():
            Ap_file = opj(self.apertures_dir, stimulus_config.get('ap_fn'))
            self.logger.info(f"aperture file: {Ap_file}")
            if Ap_file:
                try:
                    # Load the aperture file using scipy.io.loadmat
                    mat = scipy.io.loadmat(Ap_file)
                    design_matrix = mat["stim"]
                    # Add a new field to the dictionary with the loaded array
                    stimulus_config['design_matrix'] = design_matrix
                    self.logger.info(f"Loaded aperture array for {stimulus_type} stimulus type.")
                except Exception as e:
                    self.logger.info(f"Error loading aperture file for {stimulus_type} stimulus type: {str(e)}")
            else:
                self.logger.info(f"No aperture file specified for {stimulus_type} stimulus type.")

class DataCleanConfig:
    """
    This class contains data cleaning parameters that are used in the img_utils.CleanInputData class.

    Attributes:
        detrend (bool): whether or not to detrend the data
        standardize (str): type of standardization to apply to the data
        low_pass (float): low pass filter frequency (Hz)
        high_pass (float): high pass filter frequency (Hz)
        TR (float): repetition time (s)
        confounds (str): confound regressors to be included in the model
    """

    def __init__(self, config_file, mri_config):
        self.detrend, self.standardize, self.low_pass, self.high_pass, self.filter, self.confounds = self._load_config(config_file)

        self.TR = mri_config.TR

    def _load_config(self, config_file):
        with open(config_file) as f:
            config_data = json.load(f)
        
        class_name = self.__class__.__name__
        class_section = config_data.get(class_name, {})

        self.detrend = class_section.get('detrend', True)
        self.standardize = class_section.get('standardize', 'zscore')
        self.low_pass = class_section.get('low_pass', 0.1)              # Low pass filters out high frequency signals from our data: 
                                                                        # fMRI signals are slow evolving processes, any high frequency signals 
                                                                        # are likely due to noise 
        self.high_pass = class_section.get('high_pass', 0.01)           # High pass filters out any very low frequency signals (below 0.01Hz), 
                                                                        # which may be due to intrinsic scanner instabilities
        self.filter = class_section.get('filter', 'butterworth')        # type of filter to use for bandpass filtering
        self.confounds = class_section.get('confounds', None)           # could add motion regressors here

        return self.detrend, self.standardize, self.low_pass, self.high_pass, self.filter, self.confounds