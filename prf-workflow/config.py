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
    elif isinstance(config, list):
        for i, item in enumerate(config):
            config[i] = replace_placeholders(item, replacements)
    return config

class ProjectConfig:
    """
    This class contains project-specific information.

    Attributes:
        subject_list (list): list of all subjects in the project
        subject_id (str): subject to be analyzed
        hem_list (list): list of all hemispheres to be analyzed in the project
        hemi (str): hemisphere to be analyzed
        n_surfs (int): number of surfaces to be analyzed
        logger_dir (str): directory to save the log files
        do_cf_modeling (bool): whether or not to perform CF modeling
        n_procs (int): number of cores to use
    """
    
    def __init__(self, config_file, sub_idx, hem_idx):
        # get config from config file
        self.subject_list, self.hem_list, self.n_surfs, self.logger_dir, self.do_cf_modeling = self._load_config(config_file)
        
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
        self.do_cf_modeling = class_section.get('do_cf_modeling', False)

        return self.subject_list, self.hem_list, self.n_surfs, self.logger_dir, self.do_cf_modeling

class DirConfig:
    """
    This class contains directory-related information.

    Attributes:
        FS_dir (str): Freesurfer directory
        output_dir (str): output directory
        apertures_dir (str): directory containing stimulus apertures
        surface_tools_dir (str): directory containing surface tools
        ROI_dir (str): directory containing ROI labels
    """
    def __init__(self, config_file, project_config, logger):
        self.subject_id     = project_config.subject_id
        self.project_config = project_config
        self.logger         = logger

        # get config from config file
        self.FS_dir, self.output_dir, self.apertures_dir, self.surface_tools_dir, self.ROI_dir, self.project_config.do_cf_modeling = self._load_config(config_file)

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
        self.output_dir = class_section.get('output_dir', None)
        
        # path to stimulus apertures mat files
        self.apertures_dir = class_section.get('apertures_dir', None)
        
        # path to surface tools 
        self.surface_tools_dir = class_section.get('surface_tools_dir', None)

        # path to ROI labels
        self.ROI_dir = class_section.get('ROI_dir', None)

        # check if directories exist, write errors to logger, and return directories
        if not os.path.exists(self.FS_dir):
            self.logger.error('Freesurfer directory does not exist.')
            sys.exit(1)
        else:
            self.logger.info('Freesurfer directory: ' + self.FS_dir)
        
        if not os.path.exists(self.output_dir):
            # if output dir doesn't exist, create it
            os.makedirs(self.output_dir)
            self.logger.info('Output directory created: ' + self.output_dir)
        else: 
            self.logger.info('Output directory: ' + self.output_dir)
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
        if not os.path.exists(self.ROI_dir) and self.project_config.do_cf_modeling:
                # if ROI dir doesn't exist, don't run CF modeling
                self.logger.error('ROI directory does not exist in the location specified in the config file: '+self.ROI_dir)
                self.logger.error('CF modeling will not be performed.')
                # set do_cf_modeling to False
                self.project_config.do_cf_modeling = False
        else:
            self.logger.info('ROI directory: ' + self.ROI_dir)

        return self.FS_dir, self.output_dir, self.apertures_dir, self.surface_tools_dir, self.ROI_dir, self.project_config.do_cf_modeling

class PrfMappingConfig:
    """
    This class contains pRF mapping-related information.

    Attributes:
        output_dir (str): output directory
        subject_id (str): subject to be analyzed
        hemi (str): hemisphere to be analyzed
        n_surfs (int): number of surfaces to be analyzed
        logger (logging.Logger): logger object
        screen_height_cm (float): full screen height (cm)
        screen_distance_cm (float): distance from screen (cm)
        which_model (str): pRF model type ('Iso' or 'DoG')
        avg_runs (bool): whether or not to average runs of the same aperture type
        fit_hrf (bool): whether or not to fit two extra parameters for hrf derivative and dispersion
        start_from_avg (bool): whether to use avg across depths as starting point for layer fits
        grid_nr (int): size of grid for pRF mapping initial grid search
        y_coord_cutoff (int): occipital mask y-coordinate cut-off (including only posterior vertices)
        verbose (bool): whether to print out progress messages
        hrf (str, list, or numpy.ndarray): HRF shape for this Model
        filter_predictions (bool): whether to high-pass filter the predictions
        filter_type (str): type of filter to use for high-pass filtering
        filter_params (dict): parameters for high-pass filtering
        normalize_RFs (bool): whether to normalize the RF volumes
        rsq_thresh_itfit (float): Rsq threshold for iterative fitting
        rsq_thresh_viz (float): Rsq threshold for visualization
        overwrite_viz (bool): whether to overwrite the mgh param visualization files
        reference_aperture (str): reference aperture to be used for pRF model fit
        ap_combine (str): method of combining different stimulus apertures
        concat_padding (int): how many artificial baseline volumes to add between different apertures when concatenating

    """
    def __init__(self, config_file, dir_config, project_config, logger):
        self.output_dir = dir_config.output_dir
        self.subject_id = project_config.subject_id
        self.hemi       = project_config.hemi
        self.n_surfs    = project_config.n_surfs
        self.logger     = logger
        
        # get config from config file
        self.screen_height_cm, self.screen_distance_cm, self.which_model, self.avg_runs, self.fit_hrf, self.start_from_avg, self.grid_nr, self.y_coord_cutoff, self.verbose, self.hrf, self.filter_predictions, self.filter_type, self.filter_params, self.normalize_RFs, self.rsq_thresh_itfit, self.rsq_thresh_viz, self.reference_aperture, self.overwrite_viz, self.ap_combine, self.concat_padding = \
            self._load_config(config_file)

        # calculate screen dimensions
        self._get_screen_dimensions()

        # get grid search parameters
        self.size_grid, self.ecc_grid, self.polar_grid, self.surround_amplitude_grid, self.surround_size_grid = self._get_grid_search_params()

        # initialize pRF model preferences and output directory name
        self.model_name, self.avg_runs, self.prf_output_dir = self._get_prf_mapping_config()
        
        # initialize prf output filenames
        self.input_data_dict_fn, self.output_data_dict_fn, self.pRF_param_pckl_fn, self.pRF_param_map_mgh = self._get_prf_output_fns()

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
        self.rsq_thresh_viz = class_section.get('rsq_thresh_viz', 0.1)         # float
                                                                                # Rsq threshold for visualization. Must be between 0 and 1.   
        self.overwrite_viz      = class_section.get('overwrite_viz', False)     # boolean, optional
                                                                                # whether to overwrite the mgh param visualization files
        self.reference_aperture = class_section.get('reference_aperture',  None) # if not None, the pRF model fit from this aperture will be used to initialize
                                                                                    # the fitting for other apertures
        # how to deal with multiple stimulus apertures
        self.ap_combine = class_section.get('ap_combine', 'separate') # 'concatenate' or 'separate'
                                                                                # if concatenate: aperture files and fMRI data are concatenated
                                                                                # if separate: different apertures are analyzed separately
        if self.ap_combine not in ['concatenate', 'separate', False]:
            self.logger.error('ap_combine (method of combining different stimulus apertures) must be either "concatenate" or "separate". Please check the configuration file.')
            sys.exit(1)

        if self.ap_combine == 'concatenate':
            self.concat_padding = class_section.get('concat_padding', 10) # int
                                                                          # how many artificial baseline volumes to add between different apertures when concatenating
        else:
            self.concat_padding = None

        # if not none, print out the reference aperture
        if self.reference_aperture is not None:
            self.logger.info('Selected reference aperture: '+self.reference_aperture)
                                                                                                                                    

        return self.screen_height_cm, self.screen_distance_cm, self.which_model, self.avg_runs, self.fit_hrf, self.start_from_avg, self.grid_nr, self.y_coord_cutoff, self.verbose, self.hrf, self.filter_predictions, self.filter_type, self.filter_params, self.normalize_RFs, self.rsq_thresh_itfit, self.rsq_thresh_viz, self.reference_aperture, self.overwrite_viz, self.ap_combine, self.concat_padding

    def _get_screen_dimensions(self):
        # pRF mapping stimulus dimensions
        self.screen_halfheight_cm = self.screen_height_cm/2
        self.max_ecc                 = math.atan(self.screen_halfheight_cm/self.screen_distance_cm)
        self.max_ecc_deg             = math.degrees(self.max_ecc)
        self.max_ecc_size            = math.ceil(self.max_ecc_deg)

        self.logger.info('Max eccentricity: '+str(self.max_ecc_deg)+' degrees')
    
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
        
    def _get_prf_mapping_config(self):
        # prf mapping preferences

        # model name (based on above preferences)
        if self.n_surfs > 1:
            model_name          = 'prf_'+self.which_model+'_fit_hrf_'+str(self.fit_hrf)+'_start_from_avg_'+str(self.start_from_avg)+'_n_surfs_'+str(self.n_surfs)
            prf_output_dir      = opj(self.output_dir,model_name,self.subject_id)
        else:
            self.avg_runs       = None
            model_name          = 'prf_'+self.which_model+'_fit_hrf_'+str(self.fit_hrf)
            prf_output_dir      = opj(self.output_dir,model_name,self.subject_id)
        
        # check if prf_output_dir exists, if not, create it
        if not os.path.exists(prf_output_dir):
            os.makedirs(prf_output_dir)

        return model_name, self.avg_runs, prf_output_dir
        
    def _get_prf_output_fns(self):
        # data dictionary files
        input_data_dict_fn      = opj(self.prf_output_dir,self.hemi+'_input_data.pckl')
        output_data_dict_fn     = opj(self.prf_output_dir,self.hemi+'_output_data.pckl')
        pRF_param_pckl_fn       = opj(self.prf_output_dir,self.hemi+'_pRF_params_{which_surf}{which_model}.pckl')
        
        # prf parameter maps
        pRF_param_map_mgh           = opj(self.prf_output_dir,self.hemi+'.{param_name}.{aperture_type}.{depth}.mgh')            

        return input_data_dict_fn, output_data_dict_fn, pRF_param_pckl_fn, pRF_param_map_mgh

class CfModelingConfig:
    """
    This class contains CF modeling-related information.

    Attributes:
        roi_list (list): list of source regions
        subsurfaces (dict): dictionary containing info about sub-surfaces
        target_surfs (list or string): list of target surfaces or the string "all"
        CF_sizes (array): np.array of CF sizes
        rsq_thresh_itfit (float): Rsq threshold for iterative fitting
        rsq_thresh_viz (float): Rsq threshold for visualization
        verbose (bool): whether to print out progress messages
        use_bounds (bool): whether to use bounds for the CF model fit
        use_constraints (bool): whether to use constraints for the CF model fit
        overwrite_viz (bool): whether to overwrite the mgh param visualization files
        cfm_output_dir (str): output directory for CFM
        input_data_dict_fn (str): input data dictionary filename
        output_data_dict_fn (str): output data dictionary filename
        cfm_param_fn (str): CFM parameter filename
    """
    def __init__(self, config_file, project_config, dir_config, prf_config, logger):
        self.n_surfs        = project_config.n_surfs
        self.hemi           = project_config.hemi
        self.logger         = logger
        self.prf_output_dir = prf_config.prf_output_dir
        self.ROI_dir        = dir_config.ROI_dir

        self.roi_list, self.subsurfaces, self.target_surfs, self.CF_sizes, self.rsq_thresh_itfit, self.rsq_thresh_viz, self.verbose, self.use_bounds, self.use_constraints, self.overwrite_viz = self._load_config(config_file)

        self.cfm_output_dir, self.input_data_dict_fn, self.output_data_dict_fn, self.cfm_param_fn = self._get_cf_output_fns()

    def _load_config(self, config_file):
        with open(config_file) as f:
            config_data = json.load(f)
        
        # get CFModelingConfig section from config file (if it exists):
        class_name = self.__class__.__name__
        class_section = config_data.get(class_name, {})


        self.roi_list = class_section.get('roi_list', None)
        # check that roi_list isn't empty
        if not self.roi_list:
            self.logger.error('List of source regions (roi_list) is empty. Please check the configuration file.')
            sys.exit(1)
        
        # append path and '.label' extension to each roi in roi_list
        self.roi_list = [opj(self.ROI_dir, self.hemi+'.'+roi+'.label') for roi in self.roi_list]

        # check that all files in list of roi_list exist
        if self.roi_list:
            for roi in self.roi_list:
                if not os.path.exists(roi):
                    self.logger.error('ROI file does not exist: '+roi)
                    self.logger.error('Make sure all ROI labels exist before running CF analysis!')
                    sys.exit(1)

        self.subsurfaces = class_section.get('subsurfaces', None)
        # check that subsurfaces is not empty
        if not self.subsurfaces:
            self.logger.error('Subsurfaces dictionary (subsurfaces) is empty. Please check the configuration file.')
            sys.exit(1)
        else:
            # check that each element in the dictionary contains a "roi" and "depth" key and that both are ints
            for key, value in self.subsurfaces.items():
                if not isinstance(value, dict):
                    self.logger.error('Value for '+key+' subsurface is not a dictionary. Please check the configuration file.')
                    sys.exit(1)
                # check if the subsurface contains elements "roi" and "depth"
                if 'roi' not in value or 'depth' not in value:
                    self.logger.error('Value for '+key+' subsurface does not contain a "roi" or "depth" key. Please check the configuration file.')
                    sys.exit(1)              
                if not isinstance(value['roi'], int) or not isinstance(value['depth'], int):
                    self.logger.error('Value for '+key+' subsurface contains a "roi" or "depth" key that is not an integer. Please check the configuration file.')
                    sys.exit(1)
                # check that each "roi" key is smaller than or equal to the length of roi_list
                if value['roi'] > len(self.roi_list)-1:
                    self.logger.error('Value for '+key+' subsurface contains a "roi" key that is greater than the length of roi_list. Please check the configuration file.')
                    sys.exit(1)
                # and that each "depth" key is smaller than or equal to n_surfs-1
                if value['depth'] > self.n_surfs-1:
                    self.logger.error('Value for '+key+' subsurface contains a "depth" key that is greater than n_surfs-1. Please check the configuration file.')
                    sys.exit(1)
        
        self.target_surfs = class_section.get('target_surfs', 'all')
        # check that target_surfs is either a list of ints or the string "all"
        if not isinstance(self.target_surfs, list) and self.target_surfs != "all":
            self.logger.error('target_surfs must be either a list of integers (corresponding to cortical depths) or the string "all". Please check the configuration file.')
            sys.exit(1)
        # if target_surfs is a list of ints, check that none of the ints are greater than n_surfs
        if isinstance(self.target_surfs, list):
            for surf in self.target_surfs:
                if surf > self.n_surfs-1:
                    self.logger.error('current surf: '+str(surf))
                    self.logger.error('Value in list of target surfaces (target_surfs) cannot be greater than n_surfs-1. Please check the configuration file.')
                    sys.exit(1)
                # also check that none of the ints are smaller than 0
                if surf < 0:
                    self.logger.error('Value in list of target surfaces (target_surfs) cannot be smaller than 0. Please check the configuration file.')
                    sys.exit(1)
        # if target_surfs is 'all', change it to a list of all cortical depths instead
        if self.target_surfs == 'all':
            self.target_surfs = list(range(self.n_surfs))
                
        self.CF_sizes = class_section.get('CF_sizes', None)
        # check that CF_sizes is not empty
        if not self.CF_sizes:
            self.logger.error('CF_sizes is empty. Please check the configuration file.')
            sys.exit(1) 
        # check that CF_sizes is an np.array, if not, convert it to one
        if not isinstance(self.CF_sizes, np.ndarray):
            self.CF_sizes = np.array(self.CF_sizes)   

        # Input parameters for model fit
        self.rsq_thresh_itfit   = class_section.get('rsq_thresh_itfit', 0.1)    # float
                                                                                # Rsq threshold for iterative fitting. Must be between 0 and 1.         
        self.rsq_thresh_viz     = class_section.get('rsq_thresh_viz', 0.1)      # float
                                                                                # Rsq threshold for visualization. Must be between 0 and 1. 
        self.overwrite_viz      = class_section.get('overwrite_viz', False)     # boolean, optional
                                                                                # whether to overwrite the mgh param visualization files
        self.verbose            = class_section.get('verbose', True)            # boolean, optional
                                                                                # whether to print out progress messages   
        self.use_bounds         = class_section.get('use_bounds', True)         # boolean, optional
                                                                                # whether to use bounds for the CF model fit
        self.use_constraints    = class_section.get('use_constraints', True)    # boolean, optional
                                                                                # whether to use constraints for the CF model fit
        return self.roi_list, self.subsurfaces, self.target_surfs, self.CF_sizes, self.rsq_thresh_itfit, self.rsq_thresh_viz, self.verbose, self.use_bounds, self.use_constraints, self.overwrite_viz
    
    def _get_cf_output_fns(self):

        cfm_output_dir = opj(self.prf_output_dir,'cf')

        # check if cfm_output_dir exists, if not, create it
        if not os.path.exists(cfm_output_dir):
            os.makedirs(cfm_output_dir)

        # data dictionary files
        input_data_dict_fn      = opj(cfm_output_dir,self.hemi+'_input_data.pckl')
        output_data_dict_fn     = opj(cfm_output_dir,self.hemi+'_output_data.pckl')

        # prfpy outputs
        cfm_param_fn      = opj(cfm_output_dir,self.hemi+'_cfm_params.pckl')

        return cfm_output_dir, input_data_dict_fn, output_data_dict_fn, cfm_param_fn

class MriConfig:
    """
    This class contains MRI acquisition-related information.

    Attributes:
        TR (float): repetition time
        equivol_fn (str): equivolumetric surface filename prefix
        meanFunc_nii_fn (str): mean functional nitfti filepath and name
        prf_run_config (dict): dictionary containing info about pRF runs
        cf_run_config (dict): dictionary containing info about CFM runs
        gm_surf_fn (str): Freesurfer gray matter surface filename
        wm_surf_fn (str): Freesurfer white matter surface filename
        inflated_surf_fn (str): Freesurfer inflated surface filename
        equi_surf_fn_list (list): list of equivolumetric surface filenames
        meanFunc_mgh_fn (str): surface-projected mean functional run filename
        occ_mask_fn (str): occipital mask filename
    """

    def __init__(self, config_file, project_config, dir_config, prf_config, logger, cfm_config=None):
        self.prf_output_dir     = prf_config.prf_output_dir
        self.reference_aperture = prf_config.reference_aperture
        self.ap_combine         = prf_config.ap_combine
        self.concat_padding     = prf_config.concat_padding
        self.FS_dir             = dir_config.FS_dir
        self.subject_id         = project_config.subject_id
        self.hemi               = project_config.hemi
        self.n_surfs            = project_config.n_surfs
        self.do_cf_modeling     = project_config.do_cf_modeling
        self.logger             = logger

        # get config from config file
        if self.do_cf_modeling and cfm_config is not None:
            self.cfm_config = cfm_config
            self.TR, self.equivol_fn, self.meanFunc_nii_fn, self.prf_run_config, self.cf_run_config = self._load_config(config_file)
        else:
            self.TR, self.equivol_fn, self.meanFunc_nii_fn, self.prf_run_config = self._load_config(config_file)
            self.cf_run_config = None

        # get input mri filenames
        self.gm_surf_fn, self.wm_surf_fn, self.inflated_surf_fn, self.cort_label_fn, self.equi_surf_fn_list, self.meanFunc_mgh_fn, self.occ_mask_fn = self._get_mri_fns()

        # get info about pRF runs (apertures, nr sessions per run, filenames of projected runs)
        self.prf_run_config, self.prfpy_output_config, self.ap_combine = self._get_prf_run_list()

        # get info about CFM runs (nr sessions per run, filenames of projected runs)
        if self.do_cf_modeling and cfm_config is not None:
            self.cf_run_config, self.cfm_output_config = self._get_cf_run_config()
        else:
            self.cf_run_config, self.cfm_output_config = None, None

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
            sys.exit(1)
        else:
            self.logger.info('Mean functional image: ' + self.meanFunc_nii_fn)
            
        for aperture_type, config in self.prf_run_config.items():
            for run in range(config['n_runs']):
                if not os.path.exists(config['nii_fn_list'][run]):
                    self.logger.error('PRF '+aperture_type+' run '+str(run)+' does not exist under this address: '+config['nii_fn_list'][run])
                    sys.exit(1)
                else:
                    self.logger.info('PRF '+aperture_type+' run '+str(run)+': '+config['nii_fn_list'][run])

        # Check that the reference aperture is among the apertures provided in prf_run_config
        if (self.reference_aperture is not None) and (self.reference_aperture not in self.prf_run_config):
            self.logger.error('The selected reference aperture ('+self.reference_aperture+') is not present in the list of all stimulus apertures.')
            sys.exit(1)

        # Load CFM config (if applicable)
        if self.do_cf_modeling:
            self.cf_run_config  = class_section.get('cf_run_config',  None) # dictionary containing info about CFM runs

            # Check that CFM nifti files exist  
            for aperture_type, config in self.cf_run_config.items():
                for run in range(config['n_runs']):
                    if not os.path.exists(config['nii_fn_list'][run]):
                        self.logger.error('CFM '+aperture_type+' run '+str(run)+' does not exist under this address: '+config['nii_fn_list'][run])
                        sys.exit(1)
                    else:
                        self.logger.info('CFM '+aperture_type+' run '+str(run)+': '+config['nii_fn_list'][run]) 

        if self.do_cf_modeling:
            return self.TR, self.equivol_fn, self.meanFunc_nii_fn, self.prf_run_config, self.cf_run_config
        else:
            return self.TR, self.equivol_fn, self.meanFunc_nii_fn, self.prf_run_config
    
    def _get_mri_fns(self):
        # Freesurfer mesh filenames
        gm_surf_fn          = opj(self.FS_dir,self.subject_id,'surf',self.hemi+'.pial')
        wm_surf_fn          = opj(self.FS_dir,self.subject_id,'surf',self.hemi+'.white')
        inflated_surf_fn    = opj(self.FS_dir,self.subject_id,'surf',self.hemi+'.inflated')
        cort_label_fn       = opj(self.FS_dir,self.subject_id,'label',self.hemi+'.cortex.label')

        # equivolumetric surface output filenames
        if self.n_surfs > 1:
            equi_surf_fn_list = []
            for depth in range(self.n_surfs):         
                equi_surf_fn_list.append(self.equivol_fn+'{}.pial'.format(str(float(depth)/(self.n_surfs-1))))
            equi_surf_fn_list.reverse() # reverse order of surfaces so that wm (lh.equi1.0.pial) is first
        else:
            equi_surf_fn_list = None

        # surface-projected functional run filenames
        meanFunc_mgh_fn  = opj(self.prf_output_dir,self.hemi+'.meanFunc.mgh')

        # occipital mask filename
        occ_mask_fn             = opj(self.prf_output_dir,self.hemi+'_occ_mask.pckl')

        return gm_surf_fn, wm_surf_fn, inflated_surf_fn, cort_label_fn, equi_surf_fn_list, meanFunc_mgh_fn, occ_mask_fn
    
    def _get_prf_run_list(self):
        for aperture_type, config in self.prf_run_config.items():
            mgh_fn_list = []
            for run in range(config['n_runs']):
                mgh_fn = []
                for depth in range(self.n_surfs if self.n_surfs > 1 else 1):
                    mgh_fn.append(opj(self.prf_output_dir, f"{self.hemi}.equi{depth}.{aperture_type}{run + 1}.mgh"))
                mgh_fn_list.append(mgh_fn)

            config['mgh_fn_list'] = mgh_fn_list

        aperture_types = self.prf_run_config.keys()
        if self.ap_combine == 'separate' or self.ap_combine is None or len(aperture_types) == 1:
            self.ap_combine = 'separate'
        elif self.ap_combine == 'concatenate':
            aperture_types = ['combined']

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
            } for key in aperture_types
        }            

        return self.prf_run_config, prfpy_output_config, self.ap_combine
    
    def _get_cf_run_config(self):
        for aperture_type, config in self.cf_run_config.items():
            mgh_fn_list = []
            for run in range(config['n_runs']):
                mgh_fn = []
                for depth in range(self.n_surfs if self.n_surfs > 1 else 1):
                    mgh_fn.append(opj(self.prf_output_dir, f"{self.hemi}.equi{depth}.{aperture_type}{run + 1}.mgh"))
                mgh_fn_list.append(mgh_fn)

            config['mgh_fn_list'] = mgh_fn_list


        cfm_output_config = {
            key: {
                subsurf: {
                    'roi_label': self.cfm_config.roi_list[self.cfm_config.subsurfaces[subsurf]['roi']],
                    'subsurface': [],
                    'depth': self.cfm_config.subsurfaces[subsurf]['depth'],
                    'surf_fn': opj(self.FS_dir,self.subject_id, 'surf', self.hemi+'.'+self.equi_surf_fn_list[self.cfm_config.subsurfaces[subsurf]['depth']]),
                    'surf': [],
                    'dist': [],
                    'subsurface_translated': [],
                    'data': [],
                    'stim': [],
                    'model': [],
                    'gf': [],
                    'is_gf': {
                        'gridfit': False,
                        'itfit': False
                    }
                        
                } for subsurf in self.cfm_config.subsurfaces
            } for key in self.cf_run_config
        }

        return self.cf_run_config, cfm_output_config

class StimApertureConfig:
    """
    This class loads the stimulus apertures into the prf_run_config dictionary.

    Attributes:
        prf_run_config (dict): dictionary containing info about pRF runs
        apertures_dir (str): directory containing stimulus apertures
        logger (logging.Logger): logger object

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
                    design_matrix = mat["aperture"]
                    # Add a new field to the dictionary with the loaded array
                    stimulus_config['design_matrix'] = design_matrix
                    self.logger.info(f"Loaded aperture array for {stimulus_type} stimulus type.")
                except Exception as e:
                    self.logger.error(f"Error loading aperture file for {stimulus_type} stimulus type: {str(e)}")
                    sys.exit(1)
            else:
                self.logger.error(f"No aperture file specified for {stimulus_type} stimulus type.")
                sys.exit(1)

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