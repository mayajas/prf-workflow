# config.py

import os
from os.path import join as opj
import scipy
import math
import numpy as np
import logging

class ProjectConfig:
    """
    This class contains project-specific information.

    Attributes:
        proj_id (str): name of the project
        subject_list (list): list of all subjects in the project
        hem_list (list): list of all hemispheres to be analyzed in the project
    """
    # project name
    proj_id                 = 'example-project'

    # subjects and hemispheres
    subject_list            = ['sub-01','sub-02','sub-03','sub-04']
    hem_list                = ['lh','rh']

    n_surfs         = 6         # number of equivolumetric surfaces to use (including pial and white)
                                # if n_surfs = 1, then analysis is run on pial surface only

    def __init__(self, sub_idx, hem_idx, run_locally):
        self.subject_id      = self.subject_list[sub_idx]
        self.hemi            = self.hem_list[hem_idx]
        self.run_locally     = run_locally

        # Determine the number of cores
        self.n_procs = 1 if run_locally else int(os.getenv('OMP_NUM_THREADS'))

class DirConfig:
    def __init__(self, project_config, logger):
        self.run_locally    = project_config.run_locally
        self.proj_id        = project_config.proj_id
        self.subject_id     = project_config.subject_id
        self.logger         = logger

        self.proj_dir, self.home_dir, self.programs_dir, self.prf_input_dir, self.FS_dir, self.apertures_dir, self.surface_tools_dir, self.working_dir = self._get_project_dirs()
    
    def _get_project_dirs(self):
        # project, home and programs directories
        if self.run_locally:
            proj_dir                = '/home/mayajas/scratch/'+self.proj_id+'/'
            home_dir                = '/home/mayajas/Documents/'+self.proj_id+'/'
            programs_dir            = '/home/mayajas/Documents/programs/'
        else:
            proj_dir                = '/scratch/mayaaj90/'+self.proj_id+'/'
            home_dir                = '/home/mayaaj90/projects/'+self.proj_id+'/'
            programs_dir            = '/home/mayaaj90/programs/'
        self.logger.info(f"proj_dir: {proj_dir}")
        self.logger.info(f"home_dir: {home_dir}")
        self.logger.info(f"programs_dir: {programs_dir}")

        # prepped input directory
        prf_input_dir = opj(proj_dir,'output','prfpy_surf_prepped_inputs',self.subject_id)
        self.logger.info(f"prf_input_dir: {prf_input_dir}")

        # freesurfer directory
        FS_dir          = opj(proj_dir,'derivatives','wf_advanced_skullstrip',
                        '_subject_id_'+self.subject_id,'autorecon_pial')
        self.logger.info(f"FS_dir: {FS_dir}")
            
        # path to stimulus apertures mat files
        apertures_dir = opj(home_dir,'code','stim-scripts','apertures')
        self.logger.info(f"apertures_dir: {apertures_dir}")
            
        # path to surface tools 
        surface_tools_dir   = opj(programs_dir,'surface_tools','equivolumetric_surfaces')
        self.logger.info(f"surface_tools_dir: {surface_tools_dir}")

        ## Change working directory if needed
        working_dir = opj(home_dir, 'code', 'analysis-scripts', 'python')
        if os.getcwd() != working_dir:
            os.chdir(working_dir)

        return proj_dir, home_dir, programs_dir, prf_input_dir, FS_dir, apertures_dir, surface_tools_dir, working_dir

class PrfMappingConfig:
    """
    This class contains pRF mapping-related information.
    """
    # pRF mapping stimulus dimensions
    screen_height_cm        = 12.00                     # full screen height (cm)
    screen_halfheight_cm    = screen_height_cm/2        # half screen size (cm)
    screen_distance_cm      = 52.0                      # distance from screen (cm)
    max_ecc                 = math.atan(screen_halfheight_cm/screen_distance_cm)
    max_ecc_deg             = math.degrees(max_ecc)
    max_ecc_size            = round(max_ecc_deg,2)
    
    # pRF model preferences
    which_model     = 'DoG'     # 'Iso' or 'DoG'
    avg_runs        = True      # boolean
                                # whether or not to average runs of the same aperture type
                                # if False, then runs are concatenated
    fit_hrf         = False     # boolean
                                # whether or not to fit two extra parameters for hrf derivative and
                                # dispersion.
    start_from_avg  = True      # boolean
                                # whether to use avg across depths as starting point for layer fits
                                # when not fitting layer-specific prfs, this parameter is meaningless
                                # as prf runs are projected to the gm surface and fitting is done there (only once)
    fit_css         = False     # boolean
                                # TODO: add description

    # size of grid for pRF mapping initial grid search
    grid_nr         = 30

    # occipital mask y-coordinate cut-off (including only posterior vertices)
    y_coord_cutoff  = -25       # careful when choosing this - use inflated surface on CURTA in jupyter notebook to check

    # Input parameters of Iso2DGaussianModel
    verbose             = True
    hrf                 = None  # string, list or numpy.ndarray, optional
                                # HRF shape for this Model.
                                # Can be 'direct', which implements nothing (for eCoG or later convolution),
                                # a list or array of 3, which are multiplied with the three spm HRF basis functions,
                                # and an array already sampled on the TR by the user.
                                # (the default is None, which implements standard spm HRF)
    filter_predictions  = False # boolean, optional
                                # whether to high-pass filter the predictions, default False
    filter_type         = 'sg'

    sg_filter_window_length = 201
    sg_filter_polyorder     = 3

    filter_params           = {'window_length':sg_filter_window_length, 
                                'polyorder':sg_filter_polyorder}
    normalize_RFs           = False    # whether or not to normalize the RF volumes (generally not needed).

    # Input parameters for iterative fit
    rsq_thresh_itfit    = 0.0005      # float, Rsq threshold for iterative fitting. Must be between 0 and 1.

    rsq_thresh_viz      = 0.2         # float, Rsq threshold for visualization. Must be between 0 and 1.

    def __init__(self, dir_config, project_config, logger):
        self.proj_dir   = dir_config.proj_dir
        self.subject_id = project_config.subject_id
        self.hemi       = project_config.hemi
        self.n_surfs    = project_config.n_surfs
        self.logger     = logger

        # get grid search parameters
        self.size_grid, self.ecc_grid, self.polar_grid, self.surround_amplitude_grid, self.surround_size_grid = self._get_grid_search_params()

        # initialize pRF model preferences and output directory name
        self.model_name, self.avg_runs, self.out_dir = self._get_prf_mapping_config()
        
        # initialize prf output filenames
        self.input_data_dict_fn, self.output_data_dict_fn, self.pRF_param_avg_fn, self.x_map_mgh, self.y_map_mgh, self.prf_size_map_mgh, self.prf_amp_map_mgh, self.bold_baseline_map_mgh, self.srf_amp_map_mgh, self.srf_size_map_mgh, self.hrf_1_map_mgh, self.hrf_2_map_mgh, self.rsq_map_mgh, self.polar_map_mgh, self.ecc_map_mgh, self.pRF_param_per_depth_fn, self.polar_map_per_depth_mgh, self.ecc_map_per_depth_mgh, self.hrf_1_map_per_depth_mgh, self.hrf_2_map_per_depth_mgh = self._get_prf_output_fns()
         
   
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
            out_dir        = opj(self.proj_dir,'output',model_name,self.subject_id)
        else:
            self.start_from_avg = None
            model_name          = 'prf_'+self.which_model+'_fit_hrf_'+str(self.fit_hrf)
            out_dir             = opj(self.proj_dir,'output',model_name,self.subject_id)
        
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
    TR              = 3.0           # repetition time (s)
    equivol_fn      = 'equi'        # equivolumetric surface filename prefix

    prf_run_config  = {
        'bar': {
            'n_runs': 2,
            'ap_fn': 'stimulus_bar.mat',
            'fn_prefix': 'reg_bar',
            'nii_fn_list': [],
            'mgh_fn_list': []
        }#,
        # 'wedge': {
        #     'n_runs': 2,
        #     'ap_fn': 'stimulus_wedge.mat',
        #     'fn_prefix': 'reg_wedge',
        #     'nii_fn_list': [],
        #     'mgh_fn_list': []

        # }
    }

    def __init__(self, project_config, dir_config, prf_config, logger):
        self.prf_input_dir  = dir_config.prf_input_dir
        self.prf_output_dir = prf_config.out_dir
        self.FS_dir         = dir_config.FS_dir
        self.subject_id     = project_config.subject_id
        self.hemi           = project_config.hemi
        self.n_surfs        = project_config.n_surfs
        self.logger         = logger

        # get input mri filenames
        self.meanFunc_nii_fn, self.T1_nii_fn, self.gm_surf_fn, self.wm_surf_fn, self.inflated_surf_fn, self.equi_surf_fn_list, self.meanFunc_mgh_fn, self.occ_mask_fn = self._get_mri_fns()

        # get info about pRF runs (apertures, nr sessions per run, filenames of projected runs)
        self.prf_run_config, self.prfpy_output_config = self._get_prf_run_list()
    
    def _get_mri_fns(self):
        # mean functional
        meanFunc_nii_fn     = opj(self.prf_input_dir,'reg_meanFunc.nii')

        # anatomical image
        T1_nii_fn           = opj(self.prf_input_dir,'T1_out.nii')

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

        return meanFunc_nii_fn, T1_nii_fn, gm_surf_fn, wm_surf_fn, inflated_surf_fn, equi_surf_fn_list, meanFunc_mgh_fn, occ_mask_fn
    
    def _get_prf_run_list(self):
        for aperture_type, config in self.prf_run_config.items():
            nii_fn_list = []
            mgh_fn_list = []
            for run in range(config['n_runs']):
                nii_fn = opj(self.prf_input_dir, config['fn_prefix'] + str(run + 1) + '.nii')
                mgh_fn = []
                for depth in range(self.n_surfs if self.n_surfs > 1 else 1):
                    mgh_fn.append(opj(self.prf_output_dir, f"{self.hemi}.equi{depth}.{aperture_type}{run + 1}.mgh"))
                nii_fn_list.append(nii_fn)
                mgh_fn_list.append(mgh_fn)

            config['nii_fn_list'] = nii_fn_list
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
   
    detrend     = True
    standardize = 'zscore'
    low_pass    = 0.1           # Low pass filters out high frequency signals from our data: 
                                # fMRI signals are slow evolving processes, any high frequency signals 
                                # are likely due to noise 
    high_pass   = 0.01          # High pass filters out any very low frequency signals (below 0.01Hz), 
                                # which may be due to intrinsic scanner instabilities
    
    filter      ='butterworth'  # 'butterworth' 
    confounds   = None          # could add motion regressors here

    def __init__(self, mri_config):
        self.TR = mri_config.TR
