# img_utils.py
from fsl.data.freesurfer import loadVertexDataFile
from nilearn import surface, signal
import nipype.interfaces.freesurfer as fs
import nibabel as nib
import numpy as np
import os
from os.path import join as opj
import pickle
import subprocess
import sys
from multiprocessing import Pool, Manager, Lock
from functools import partial
import surfdist as sd
from surfdist import analysis
from joblib import Parallel, delayed

def is_running_on_slurm():
    return "SLURM_JOB_ID" in os.environ

def translate_indices(original_mask, new_mask, depth, target_surfs):
    """Used to translate the indices of the ROI mask to indices of the occipital mask, across depths"""
    mapping = {}  # Mapping from original indices to new indices
    new_indices = []  # Translated indices from the new array

    # check if new_mask is a subset of original_mask
    if not set(new_mask).issubset(set(original_mask)):
        # remove the indices that are not in the original mask
        new_mask_orig = new_mask
        original_mask_set = set(original_mask)
        new_mask = [x for x in new_mask if x in original_mask_set]

        # write the removed index values to the flag str
        flag_str = 'The ROI mask is not a subset of the occipital mask. Removing the following indices from the subsurface: ' + ', '.join([str(idx) for idx in set(new_mask_orig).difference(set(new_mask))])    

        # write the index of the non-occipital vertices to the non_occ_vtx variable
        non_occ_vtx = np.where(np.isin(new_mask_orig, np.setdiff1d(new_mask_orig, new_mask)))[0]
    else:
        flag_str = None

        non_occ_vtx = None

    # Create a mapping between original indices and new indices
    for i, idx in enumerate(original_mask):
        mapping[idx] = i

    # Translate indices in the new mask
    for idx in new_mask:
        new_indices.append(mapping[idx])

    # If depth > 0, then add the number of vertices in the previous depth to the new indices
    if depth > 0:
        new_indices = [idx + len(original_mask) * target_surfs.index(depth) for idx in new_indices]

    return new_indices, non_occ_vtx, flag_str

def calculate_distance(src, subsurface, cort):
    """
    Function to calculate distance for a single source vertex.
    """
    wb_dist = sd.analysis.dist_calc(subsurface['surf'], cort, src)
    return wb_dist[subsurface['subsurface']]

class EquivolumetricSurfaces:
    """
    This class contains functions that are used to generate equivolumetric surfaces.

    Attributes:
        subject_id (str): Subject ID
        hemi (str): Hemisphere
        dir_config (DirConfig): Directory configuration
        mri_config (MriConfig): MRI configuration
        n_surfs (int): Number of equivolumetric surfaces
        logger (Logger): Logger
        
    """
    def __init__(self, project_config, dir_config, mri_config, logger):
        self.subject_id     = project_config.subject_id
        self.hemi           = project_config.hemi
        self.dir_config     = dir_config
        self.mri_config     = mri_config
        self.n_surfs        = project_config.n_surfs
        self.logger         = logger
        projection_surface  = mri_config.project_surf

        if self.n_surfs > 1:
            # Create full file paths for the equivolumetric surfaces
            equivol_path    = [opj(dir_config.FS_dir,self.subject_id,'surf', self.hemi+'.'+equi_surf_fn) for equi_surf_fn in self.mri_config.equi_surf_fn_list]

            # Check if equivolumetric surfaces exist
            all_files_exist = all(os.path.exists(file_path) for file_path in equivol_path)

            # If equivolumetric surfaces don't exist, generate them
            if all_files_exist:
                self.logger.info('Equivolumetric surfaces already exist.')
            else:
                self.logger.info('Generating {} equivolumetric surfaces...'.format(self.n_surfs))
                self._gen_equivol_surfs()
        else:
            self.logger.info('Analysis will be run on single surface: {}'.format(projection_surface))

    def _gen_equivol_surfs(self):
        """
        This script generates a defined number of equivolumetric surfaces from the gray to the white matter surface (Freesurfer).
        """    
        # get variables from config files
        subject_id              = self.dir_config.subject_id
        FS_dir                  = self.dir_config.FS_dir
        surface_tools_dir       = self.dir_config.surface_tools_dir
        gm_surf_fn              = self.mri_config.gm_surf_fn
        wm_surf_fn              = self.mri_config.wm_surf_fn
        n_surfs                 = self.n_surfs
        equivol_output_prefix   = self.hemi+'.'+self.mri_config.equivol_fn

        # add path to surface_tools
        sys.path.append(surface_tools_dir)

        # set environment FS subjects dir
        os.environ["SUBJECTS_DIR"] = FS_dir

        # command to generate defined nr of equivolumetric surfaces for current subject
        command = ['python', surface_tools_dir+'/generate_equivolumetric_surfaces.py',gm_surf_fn, wm_surf_fn, str(n_surfs), equivol_output_prefix,
                '--software', 'freesurfer','--smoothing','0','--subject_id',subject_id]
        self.logger.info('Command: {}'.format(' '.join(command)))

        # pass the current environment to the subprocess
        env = os.environ.copy()

        # if running on CURTA, add conda environment to path (otherwise, the nibabel package isn't found within the subprocess)
        if is_running_on_slurm():
            self.logger.info('Running on CURTA. Adding conda environment to path...')
            conda_python_path = opj(sys.prefix, 'bin')
            conda_site_packages = opj(sys.prefix, 'lib', 'python' + sys.version[:3], 'site-packages')

            command = [conda_python_path + '/' + command[0]] + command[1:]

            env['PYTHONPATH'] = conda_site_packages

        # print the command
        self.logger.info('Command: {}'.format(' '.join(command)))

        # run the command
        try:
            result = subprocess.run(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False, check=True)
            self.logger.info('Command output:\n{}'.format(result.stdout.decode('utf-8')))
            self.logger.info('Equivolumetric surfaces generated.')
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Command failed with return code {e.returncode}: {e}")
            self.logger.error('Command stderr:\n{}'.format(e.stderr.decode('utf-8')))
            sys.exit(1)  # Exit with a non-zero code to indicate error

class SurfaceProject:

    """
    This class contains functions that are used to surface-project functional data.
    """

    def __init__(self, project_config, dir_config, mri_config, logger, cfm_config=None):
        self.subject_id     = project_config.subject_id
        self.hemi           = project_config.hemi
        self.dir_config     = dir_config
        self.mri_config     = mri_config
        self.prf_run_config = mri_config.prf_run_config
        if project_config.do_cf_modeling:
            self.cf_run_config = mri_config.cf_run_config
        self.n_surfs        = project_config.n_surfs
        self.logger         = logger

        # surface project mean functionals
        source_file = self.mri_config.meanFunc_nii_fn
        if self.n_surfs > 1:
            projection_surface     = self.mri_config.equi_surf_fn_list[0]
        else:
            projection_surface     = self.mri_config.project_surf
        out_file    = self.mri_config.meanFunc_mgh_fn
        if not os.path.exists(out_file):
            logger.info('Surface-projecting mean functional...')
            logger.info('Source file: {}'.format(source_file))
            logger.info('Surface: {}'.format(projection_surface))
            logger.info('Output file: {}'.format(out_file))
            if not os.path.exists(out_file):
                try:
                    logger.info('FS_dir: {}'.format(self.dir_config.FS_dir))
                    self._surface_project(self.dir_config.FS_dir,self.subject_id,self.hemi,source_file,self.mri_config.interp_method,projection_surface,out_file)
                    logger.info('Surface-projecting mean functional completed.')
                except Exception as e:
                    logger.error(f"Error: {str(e)}")
                    logger.exception("Full exception traceback:")
                    sys.exit(1)
        else:
            logger.info('Mean functional already surface-projected.')

        # surface project pRF mapping runs (iterating over stimuli, runs and equivolumetric surface depth)
        logger.info('Surface-projecting pRF mapping runs...')
        for aperture_type, config in self.prf_run_config.items():
            logger.info('Aperture type: {}'.format(aperture_type))
            for run in range(0,config['n_runs']):
                for depth in range(0,self.n_surfs):
                    source_file = config['nii_fn_list'][run]
                    if self.n_surfs > 1:
                        projection_surface = self.mri_config.equi_surf_fn_list[depth]
                    elif depth == 0 and self.n_surfs == 1:
                        projection_surface = self.mri_config.project_surf
                    out_file = config['mgh_fn_list'][run][depth]
                    if not os.path.exists(out_file):
                        logger.info('Source file: {}'.format(source_file))
                        logger.info('Surface: {}'.format(projection_surface))
                        logger.info('Output file: {}'.format(out_file))
                        try:
                            self._surface_project(self.dir_config.FS_dir,self.subject_id,self.hemi,source_file,self.mri_config.interp_method,projection_surface,out_file)
                            logger.info('Surface-projecting pRF mapping run {} of {} for {} aperture type and {} depth completed.'.format(run+1,config['n_runs'],aperture_type,depth+1))
                        except Exception as e:
                            logger.error(f"Error: {str(e)}")
                            logger.exception("Full exception traceback:")
                            sys.exit(1)
                    else:
                        logger.info('run {} of {} for depth {} already surface-projected.'.format(run+1,config['n_runs'],depth+1))

        ## Surface-project CF mapping runs (if applicable)
        if project_config.do_cf_modeling:
            logger.info('Surface-projecting CF mapping runs...')
            for aperture_type, config in self.cf_run_config.items():
                logger.info('Aperture type: {}'.format(aperture_type))
                for run in range(0,config['n_runs']):
                    for depth in range(0,self.n_surfs):
                        source_file = config['nii_fn_list'][run]
                        if self.n_surfs > 1:
                            projection_surface = self.mri_config.equi_surf_fn_list[depth]
                        elif depth == 0 and self.n_surfs == 1:
                            projection_surface = self.mri_config.project_surf
                        out_file = config['mgh_fn_list'][run][depth]
                        if not os.path.exists(out_file):
                            logger.info('Source file: {}'.format(source_file))
                            logger.info('Surface: {}'.format(projection_surface))
                            logger.info('Output file: {}'.format(out_file))
                            try:
                                self._surface_project(self.dir_config.FS_dir,self.subject_id,self.hemi,source_file,self.mri_config.interp_method,projection_surface,out_file)
                                logger.info('Surface-projecting CF mapping run {} of {} for {} aperture type and {} depth completed.'.format(run+1,config['n_runs'],aperture_type,depth+1))
                            except Exception as e:
                                logger.error(f"Error: {str(e)}")
                                logger.exception("Full exception traceback:")
                                sys.exit(1)
                        else:
                            logger.info('run {} of {} for depth {} already surface-projected.'.format(run+1,config['n_runs'],depth+1))                

    def _surface_project(self,FS_dir,subject_id,hemi,source_file,interp_method,projection_surface,out_file):    
        # set environment FS subjects dir
        os.environ["SUBJECTS_DIR"] = FS_dir

        sampler = fs.SampleToSurface(hemi=hemi)
        sampler.inputs.source_file = source_file
        sampler.inputs.reg_header = True
        sampler.inputs.subjects_dir = FS_dir
        sampler.inputs.subject_id = subject_id
        sampler.inputs.sampling_method = "point"
        sampler.inputs.sampling_range = 0.0
        sampler.inputs.sampling_units = "mm"
        sampler.inputs.interp_method = interp_method
        # if projection_surface is a string, use it as the surface name, otherwise if a float: use projfrac
        if isinstance(projection_surface, str):
            sampler.inputs.surface = projection_surface
        elif isinstance(projection_surface, float):
            sampler.inputs.args = "--projfrac "+str(projection_surface)
        sampler.inputs.out_file = out_file
        sampler.run()

class CleanInputData:
    """
    This class contains functions that are used to clean input data.
    """
    def __init__(self, project_config, prf_config, mri_config, data_clean_config, logger, cfm_config=None):
        self.n_surfs            = project_config.n_surfs
        self.ap_combine         = prf_config.ap_combine
        self.concat_padding     = prf_config.concat_padding
        self.occ_mask_fn        = mri_config.occ_mask_fn
        self.prf_run_config     = mri_config.prf_run_config
        self.prf_config         = prf_config
        if project_config.do_cf_modeling:
            self.cf_run_config  = mri_config.cf_run_config
            self.cfm_config     = cfm_config
            self.target_surfs   = cfm_config.target_surfs
        self.y_coord_cutoff     = prf_config.y_coord_cutoff
        self.logger             = logger
        
        ## Load preprocessed surface meshes
        logger.info('Loading GM/WM surface meshes...')
        self.gm_mesh        = surface.load_surf_mesh(mri_config.gm_surf_fn) 
        self.wm_mesh        = surface.load_surf_mesh(mri_config.wm_surf_fn)

        ## Load surface-projected data
        logger.info('Loading surface-projected data...')
        
        # mean functional
        self.meanFunc_mgh   = loadVertexDataFile(mri_config.meanFunc_mgh_fn)
        
        # pRF runs across all apertures, runs, depths
        for aperture_type, config in self.prf_run_config.items():
            config['raw_data'] = {}  # Add a new key 'raw_data' for each aperture type
            for run in range(0,config['n_runs']):
                config['raw_data'][run] = {}
                for depth in range(0,self.n_surfs):
                    mgh_fn = config['mgh_fn_list'][run][depth]
                    config['raw_data'][run][depth] = loadVertexDataFile(mgh_fn)
        
        # CF runs (if applicable) across all apertures, runs, depths
        if project_config.do_cf_modeling:
            for aperture_type, config in self.cf_run_config.items():
                config['raw_data'] = {}
                for run in range(0,config['n_runs']):
                    config['raw_data'][run] = {}
                    for depth in range(0,self.n_surfs):
                        mgh_fn = config['mgh_fn_list'][run][depth]
                        config['raw_data'][run][depth] = loadVertexDataFile(mgh_fn)

        ## Make occipital mask
        self.occ_mask, self.n_vtx = self._make_occipital_mask()

        ## Clean input data - for each pRF run:
        self.detrend        = data_clean_config.detrend
        self.standardize    = data_clean_config.standardize
        self.low_pass       = data_clean_config.low_pass
        self.high_pass      = data_clean_config.high_pass     
        self.TR             = data_clean_config.TR
        self.filter         = data_clean_config.filter
        self.confounds      = data_clean_config.confounds

        mri_config.prf_run_config = self._clean_data_prf()

        ## Clean input data - for each CF run (if applicable):
        if cfm_config is not None:
            mri_config.cf_run_config = self._clean_data_cfm()

        ## Combine apertures (if applicable)
        if self.ap_combine == 'concatenate':
            mri_config.prf_run_config = self._combine_apertures()

    def _make_occipital_mask(self):
        """
        This function makes an occipital mask from the mean functional data.
        """

        if not os.path.exists(self.occ_mask_fn):
            self.logger.info('Making occipital mask...')

            # get nr of vertices on pial surface
            n_vtx           = len(self.meanFunc_mgh[:])  # nr of vertices on pial surface

            # preallocate occipital mask
            occ             = np.zeros(n_vtx)

            # set vertices posterior to y_coord_cutoff to 1
            occ[self.gm_mesh.coordinates[:,1]<self.y_coord_cutoff]=1.

            # get occipital mask coordinates
            occ_mask = np.nonzero(occ)[0]

            # save occipital mask
            f = open(self.occ_mask_fn, 'wb')
            pickle.dump([occ_mask,n_vtx], f)
            f.close()
            self.logger.info('Occipital mask saved to {}'.format(self.occ_mask_fn))
        else:
            # if occipital mask already exists, load it
            self.logger.info('Occipital mask already exists at {}'.format(self.occ_mask_fn))
            self.logger.info('Loading occipital mask...')
            with open(self.occ_mask_fn, 'rb') as pickle_file:
                [occ_mask,n_vtx] = pickle.load(pickle_file)

        return occ_mask, n_vtx
    
    def _clean_data_prf(self):
        """Clean pRF data.
        Apply occipital mask to constrain analysis to occipital pole.
        Detrend, standardize, and bandpass filter each functional pRF run.
        Average over runs."""
        if not os.path.exists(self.prf_config.input_data_dict_fn):
            self.logger.info('Cleaning pRF input data...')
            for aperture_type, config in self.prf_run_config.items():
                self.logger.info('Cleaning data for {} aperture type...'.format(aperture_type))
                config['filtered_data'] = {}  # Add a new key 'filtered_data' for each aperture type
                config['filtered_data_zscore'] = {}  # Add a new key 'filtered_data' for each aperture type
                config['masked_data'] = {}  # Add a new key 'masked_data' for each aperture type
                config['preproc_data_per_depth'] = {}  # Add a new key 'preproc_data_per_depth' for each aperture type
                config['preproc_data_per_depth_zscore'] = {}  # Add a new key 'preproc_data_per_depth_zscore' for each aperture type
                config['preproc_data_avg'] = {}  # Add a new key 'preproc_data_avg' for each aperture type
                for run in range(0,config['n_runs']):
                    self.logger.info('Run {} of {}...'.format(run+1,config['n_runs']))
                    config['masked_data'][run] = {}
                    config['filtered_data'][run] = {}
                    config['filtered_data_zscore'][run] = {}
                    for depth in range(0,self.n_surfs):
                        # Apply occipital mask to constrain analysis to occipital pole
                        self.logger.info('Applying occipital mask to constrain analysis to occipital pole... Depth: {}'.format(depth))
                        config['masked_data'][run][depth] = config['raw_data'][run][depth][self.occ_mask].T

                        # Detrend, standardize, and bandpass filter each functional pRF run
                        self.logger.info('Detrending, standardizing, and bandpass filtering each functional pRF run... Depth: {}'.format(depth))
                        config['filtered_data'][run][depth] = signal.clean(config['masked_data'][run][depth],
                                                                        confounds=self.confounds,
                                                                        detrend=self.detrend, standardize=self.standardize,
                                                                        filter=self.filter, low_pass=self.low_pass, high_pass=self.high_pass,
                                                                        t_r=self.TR)
                        config['filtered_data_zscore'][run][depth] = signal.clean(config['masked_data'][run][depth],
                                                                        confounds=self.confounds,
                                                                        detrend=self.detrend, standardize='zscore',
                                                                        filter=self.filter, low_pass=self.low_pass, high_pass=self.high_pass,
                                                                        t_r=self.TR)
                        
                # Average over runs
                self.logger.info('Averaging over runs...')
                config['preproc_data_per_depth'] = [0] * self.n_surfs
                config['preproc_data_per_depth_zscore'] = [0] * self.n_surfs
                for run in config['filtered_data']:
                    for depth in range(0,self.n_surfs):
                        config['preproc_data_per_depth'][depth] += config['filtered_data'][run][depth]
                        config['preproc_data_per_depth_zscore'][depth] += config['filtered_data_zscore'][run][depth]

                for depth in range(0,self.n_surfs):
                    config['preproc_data_per_depth'][depth] = config['preproc_data_per_depth'][depth].T
                    config['preproc_data_per_depth'][depth] /= config['n_runs']
                    config['preproc_data_per_depth_zscore'][depth] = config['preproc_data_per_depth_zscore'][depth].T
                    config['preproc_data_per_depth_zscore'][depth] /= config['n_runs']

                # Average over depths
                self.logger.info('Averaging over depths...')
                config['preproc_data_avg'] = sum(config['preproc_data_per_depth']) / self.n_surfs

            ## Save cleaned data
            self.logger.info('Saving cleaned data...')
            with open(self.prf_config.input_data_dict_fn, 'wb') as pickle_file:
                pickle.dump(self.prf_run_config, pickle_file)
        else:
            self.logger.info('Cleaned data already exists at {}'.format(self.prf_config.input_data_dict_fn))
            self.logger.info('Loading cleaned data...')
            with open(self.prf_config.input_data_dict_fn, 'rb') as pickle_file:
                self.prf_run_config = pickle.load(pickle_file)

            # Check if the cleaned data matches the dimensions of the occipital mask
            for aperture_type, config in self.prf_run_config.items():
                if aperture_type == 'combined': 
                    continue
                for run in range(0,config['n_runs']):
                    for depth in range(0,self.n_surfs):
                        if config['masked_data'][run][depth].shape[1] != self.occ_mask.shape[0]:
                            self.logger.error('Cleaned data does not match the dimensions of the occipital mask.')
                            self.logger.error('It is suggested to delete the cleaned data dictionary and the occipital mask and rerun the analysis.')
                            sys.exit(1)

            # Check if the cleaned data dictionary has all the needed keys ('filtered_data', 'masked_data', 'preproc_data_per_depth', 'preproc_data_avg')
            for aperture_type, config in self.prf_run_config.items():
                if aperture_type == 'combined': 
                    continue
                for key in ['filtered_data','filtered_data_zscore', 'masked_data', 'preproc_data_per_depth','preproc_data_per_depth_zscore', 'preproc_data_avg']:
                    if key not in config:
                        self.logger.error('Cleaned data dictionary does not have the key: {}'.format(key))
                        self.logger.error('Cleaned data dictionary must have the following keys: {}'.format(['filtered_data','filtered_data_zscore', 'masked_data', 'preproc_data_per_depth','preproc_data_per_depth_zscore', 'preproc_data_avg']))
                        self.logger.error('Cleaned data dictionary has the following keys: {}'.format(list(config.keys())))
                        self.logger.error('It is suggested to delete the cleaned data dictionary and rerun the analysis.')
                        sys.exit(1)

        return self.prf_run_config
        
    def _clean_data_cfm(self):
        """Clean CF data.
        Apply occipital mask to constrain analysis to occipital pole.
        Detrend, standardize, and bandpass filter each functional CF run.
        Average over runs."""
        if not os.path.exists(self.cfm_config.input_data_dict_fn):
            self.logger.info('Cleaning CF input data...')
            for aperture_type, config in self.cf_run_config.items():
                self.logger.info('Cleaning data for {} aperture type...'.format(aperture_type))

                # Check if the current aperture type is present in the prf_run_config and if it has the same number of runs
                if aperture_type in self.prf_run_config and config['n_runs'] == self.prf_run_config[aperture_type]['n_runs']:
                    self.logger.info('Aperture type {} is present in the prf_run_config.'.format(aperture_type))

                    # Copy the preprocessed data from the prf_run_config
                    self.logger.info('Copying preprocessed data from the prf_run_config...')
                    config['masked_data'] = self.prf_run_config[aperture_type]['masked_data']
                    config['filtered_data'] = self.prf_run_config[aperture_type]['filtered_data']
                    config['preproc_data_per_depth'] = self.prf_run_config[aperture_type]['preproc_data_per_depth']
                    config['preproc_data_concatenated_depths'] = np.concatenate([ config['preproc_data_per_depth'][index] for index in self.target_surfs ],axis=0)
                
                else:
                    # if aperture_type is not in prf_run_config
                    if aperture_type not in self.prf_run_config:
                        self.logger.info('Aperture type {} is not present in the prf_run_config.'.format(aperture_type))
                    
                    self.logger.info('Cleaning data for {} aperture type...'.format(aperture_type))
                    config['filtered_data'] = {}
                    config['masked_data'] = {}
                    config['preproc_data_per_depth'] = {}
                    config['preproc_data_concatenated_depths'] = {}
                    
                    for run in range(0,config['n_runs']):
                        self.logger.info('Run {} of {}...'.format(run+1,config['n_runs']))
                        config['masked_data'][run] = {}
                        config['filtered_data'][run] = {}
                        for depth in range(0,self.n_surfs):
                            # Apply occipital mask to constrain analysis to occipital pole
                            self.logger.info('Applying occipital mask to constrain analysis to occipital pole... Depth: {}'.format(depth))
                            config['masked_data'][run][depth] = config['raw_data'][run][depth][self.occ_mask].T

                            # Detrend, standardize, and bandpass filter each functional CF run
                            self.logger.info('Detrending, standardizing, and bandpass filtering each functional CF run... Depth: {}'.format(depth))
                            config['filtered_data'][run][depth] = signal.clean(config['masked_data'][run][depth],
                                                                            confounds=self.confounds,
                                                                            detrend=self.detrend, standardize=self.standardize,
                                                                            filter=self.filter, low_pass=self.low_pass, high_pass=self.high_pass,
                                                                            t_r=self.TR)
                    
                    # Average over runs
                    self.logger.info('Averaging over runs...')
                    config['preproc_data_per_depth'] = [0] * self.n_surfs
                    for run in config['filtered_data']:
                        for depth in range(0,self.n_surfs):
                            config['preproc_data_per_depth'][depth] += config['filtered_data'][run][depth]

                    for depth in range(0,self.n_surfs):
                        config['preproc_data_per_depth'][depth] = config['preproc_data_per_depth'][depth].T
                        config['preproc_data_per_depth'][depth] /= config['n_runs']

                    # Concatenate the preprocessed data across depths
                    self.logger.info('Concatenating the preprocessed data across depths...')
                    config['preproc_data_concatenated_depths'] = np.concatenate([ config['preproc_data_per_depth'][index] for index in self.target_surfs ],axis=0)
                    
            ## Save cleaned data
            self.logger.info('Saving cleaned data...')
            with open(self.cfm_config.input_data_dict_fn, 'wb') as pickle_file:
                pickle.dump(self.cf_run_config, pickle_file)
        else:
            self.logger.info('Cleaned data already exists at {}'.format(self.cfm_config.input_data_dict_fn))
            self.logger.info('Loading cleaned data...')
            with open(self.cfm_config.input_data_dict_fn, 'rb') as pickle_file:
                self.cf_run_config = pickle.load(pickle_file)

            # Check if the cleaned data matches the dimensions of the occipital mask
            for aperture_type, config in self.cf_run_config.items():
                for run in range(0,config['n_runs']):
                    for depth in range(0,self.n_surfs):
                        if config['masked_data'][run][depth].shape[1] != self.occ_mask.shape[0]:
                            self.logger.error('Cleaned data does not match the dimensions of the occipital mask.')
                            self.logger.error('It is suggested to delete the cleaned data dictionary and the occipital mask and rerun the analysis.')
                            sys.exit(1)

            # Check if the cleaned data dictionary has all the needed keys ('filtered_data', 'masked_data', 'preproc_data_per_depth', 'preproc_data_avg')
            for aperture_type, config in self.cf_run_config.items():
                for key in ['filtered_data', 'masked_data', 'preproc_data_per_depth', 'preproc_data_concatenated_depths']:
                    if key not in config:
                        self.logger.error('Cleaned data dictionary does not have the key: {}'.format(key))
                        self.logger.error('Cleaned data dictionary must have the following keys: {}'.format(['filtered_data', 'masked_data', 'preproc_data_per_depth', 'preproc_data_concatenated_depths']))
                        self.logger.error('Cleaned data dictionary has the following keys: {}'.format(list(config.keys())))
                        self.logger.error('It is suggested to delete the cleaned data dictionary and rerun the analysis.')
                        sys.exit(1)

        return self.cf_run_config
    
    def _combine_apertures(self):
        """
        This function combines data from the various stimulus apertures by concatenating the preprocessed data across apertures.
        """
        self.logger.info('Combining data from different stimulus apertures...')
        if not os.path.exists(self.prf_config.input_data_dict_fn):
            self.logger.error('The cleaned data dictionary does not exist. Please run the data cleaning step before combining apertures.')
            sys.exit(1)
        else:
            self.logger.info('Cleaned data already exists at {}'.format(self.prf_config.input_data_dict_fn))
            self.logger.info('Loading cleaned data...')
            with open(self.prf_config.input_data_dict_fn, 'rb') as pickle_file:
                self.prf_run_config = pickle.load(pickle_file)

            self.logger.info('Combining data from different stimulus apertures...')
            self.prf_run_config_combined = {
                'combined': {
                    'design_matrix': [],
                    'preproc_data_avg': [],
                    'preproc_data_per_depth': [0] * self.n_surfs
                }
            }

            ap = 0
            for aperture_type, config in self.prf_run_config.items():
                if aperture_type == 'combined': 
                    continue
                if ap == 0:
                    self.prf_run_config_combined['combined']['design_matrix'] = config['design_matrix']
                    self.prf_run_config_combined['combined']['preproc_data_avg'] = config['preproc_data_avg']
                    for depth in range(0,self.n_surfs):
                        self.prf_run_config_combined['combined']['preproc_data_per_depth'][depth] = config['preproc_data_per_depth'][depth]
                else:
                    if self.concat_padding > 0:
                        ## add "padding" before concatenating the data from additional apertures
                        self.logger.info('Adding padding before concatenating the data from additional apertures...')
                        padding_design_matrix = np.zeros([self.prf_run_config_combined['combined']['design_matrix'].shape[0],
                                                          self.prf_run_config_combined['combined']['design_matrix'].shape[1],
                                                          self.concat_padding])
                        padding_data = np.zeros([self.prf_run_config_combined['combined']['preproc_data_avg'].shape[0],
                                                 self.concat_padding])
                        
                        self.prf_run_config_combined['combined']['design_matrix'] = np.concatenate((self.prf_run_config_combined['combined']['design_matrix'], 
                                                                                               padding_design_matrix,
                                                                                               config['design_matrix']), axis=2)
                        self.prf_run_config_combined['combined']['preproc_data_avg'] = np.concatenate((self.prf_run_config_combined['combined']['preproc_data_avg'],
                                                                                                  padding_data,
                                                                                                  config['preproc_data_avg']), axis=1)
                        for depth in range(0,self.n_surfs):
                            self.prf_run_config_combined['combined']['preproc_data_per_depth'][depth] = np.concatenate((self.prf_run_config_combined['combined']['preproc_data_per_depth'][depth],
                                                                                                                padding_data,
                                                                                                                config['preproc_data_per_depth'][depth]), axis=1)
                    else:
                        self.logger.info('Padding is set to 0. It is highly recommended to set the padding to a value greater than 0.')
                        self.prf_run_config_combined['combined']['design_matrix'] = np.concatenate((self.prf_run_config_combined['combined']['design_matrix'], 
                                                                                               config['design_matrix']), axis=2)
                        self.prf_run_config_combined['combined']['preproc_data_avg'] = np.concatenate((self.prf_run_config_combined['combined']['preproc_data_avg'],
                                                                                                  config['preproc_data_avg']), axis=1)
                        for depth in range(0,self.n_surfs):
                            self.prf_run_config_combined['combined']['preproc_data_per_depth'][depth] = np.concatenate((self.prf_run_config_combined['combined']['preproc_data_per_depth'][depth],
                                                                                                                config['preproc_data_per_depth'][depth]), axis=1)
                ap += 1

            # write prf_run_config_combined to prf_run_config
            self.prf_run_config['combined'] = self.prf_run_config_combined['combined']

            ## Save combined data
            self.logger.info('Saving combined data...')
            with open(self.prf_config.input_data_dict_fn, 'wb') as pickle_file:
                pickle.dump(self.prf_run_config, pickle_file)

            return self.prf_run_config





class CreateSubsurfaces:
    """This class contains functions that are used to generate the subsurfaces used for connective field modeling."""
    def __init__(self, project_config, mri_config, prf_config, cfm_config, logger):
        self.n_procs            = project_config.n_procs
        self.prf_config         = prf_config
        self.cf_run_config      = mri_config.cf_run_config
        self.cfm_output_config  = mri_config.cfm_output_config
        self.occ_mask_fn        = mri_config.occ_mask_fn
        self.cfm_config         = cfm_config
        self.prf_rsq_thresh     = cfm_config.prf_rsq_thresh
        self.reference_aperture = cfm_config.reference_aperture
        self.cort_label_fn      = mri_config.cort_label_fn
        self.target_surfs       = cfm_config.target_surfs
        self.logger             = logger

        ## Load occipital mask
        logger.info('Loading occipital mask...')
        with open(self.occ_mask_fn, 'rb') as pickle_file:
            [self.occ_mask,_] = pickle.load(pickle_file)

        # Create subsurfaces
        self.logger.info('Creating subsurfaces...')
        self._create_subsurfaces()

    def _create_subsurfaces(self):
        """
        This function creates the subsurfaces used for connective field modeling.
        """

        # First, try loading cfm_output_config
        if os.path.exists(self.cfm_config.output_data_dict_fn):
            self.logger.info('Subsurfaces already exist at {}'.format(self.cfm_config.output_data_dict_fn))
            self.logger.info('Loading subsurfaces...')
            with open(self.cfm_config.output_data_dict_fn, 'rb') as pickle_file:
                self.cfm_output_config = pickle.load(pickle_file)

        ## Load cortical label
        self.logger.info('Loading cortical label...')
        self.logger.info('Cortical label: {}'.format(self.cort_label_fn))
        self.cort = nib.freesurfer.read_label(self.cort_label_fn) 

        for aperture_type, config in self.cfm_output_config.items():
            self.logger.info('Creating subsurfaces for aperture type: {}'.format(aperture_type))
            for subsurf_name, subsurface in config.items():
                # if the keys ('subsurface', 'surf', 'dist', 'data') are empty, then fill them (use a.any())
                if not len(subsurface['subsurface']) or not len(subsurface['surf']) or not len(subsurface['dist']) or not len(subsurface['data']):                
                    self.logger.info('Creating subsurface: {}'.format(subsurf_name))

                    # Load data: roi_label contains vertex numbers of given subsurface, surf_fn contains current surface geometry
                    self.logger.info('Loading relevant ROI label and cortical surface...')
                    self.logger.info('ROI label: {}'.format(subsurface['roi_label']))
                    self.logger.info('Cortical surface: {}'.format(subsurface['surf_fn']))
                    self.cfm_output_config[aperture_type][subsurf_name]['subsurface']   = nib.freesurfer.io.read_label(subsurface['roi_label'])
                    self.cfm_output_config[aperture_type][subsurf_name]['surf']         = nib.freesurfer.read_geometry(subsurface['surf_fn'])

                    if self.prf_rsq_thresh is not None:
                        self.logger.info('Removing vertices with R^2 below threshold in pRF model from subsurface...')
                        # Load rsq map (output of pRF fitting)
                        self.logger.info('Loading relevant pRF data...')
                        rsq_map_mgh = self.prf_config.pRF_param_map_mgh.format(param_name='rsq',aperture_type=self.reference_aperture,depth='avg')
                        rsq_map = np.squeeze(nib.load(rsq_map_mgh).get_fdata())
                    

                        # Remove vertices with R^2 below threshold from subsurface
                        self.logger.info('Applying R^2 threshold: {}'.format(self.prf_rsq_thresh))
                        temp = self.cfm_output_config[aperture_type][subsurf_name]['subsurface']
                        self.cfm_output_config[aperture_type][subsurf_name]['subsurface'] = temp[rsq_map[self.cfm_output_config[aperture_type][subsurf_name]['subsurface']] > self.prf_rsq_thresh]

            
                    # Get number of vertices in current subsurface
                    n_vtx_sub     = self.cfm_output_config[aperture_type][subsurf_name]['subsurface'].shape[0]
                    self.logger.info('Number of vertices in current subsurface: {}'.format(n_vtx_sub))

                    # for each vertex in subsurface, get distance to all other vertices within the subsurface
                    self.logger.info('Calculating distance matrix...')
                    self.cfm_output_config[aperture_type][subsurf_name]['dist'] = np.zeros([n_vtx_sub,n_vtx_sub])
                    
                    vtx = 0
                    for src in self.cfm_output_config[aperture_type][subsurf_name]['subsurface']:
                        print(str(vtx), end="\r")
                        self.cfm_output_config[aperture_type][subsurf_name]['dist'][vtx,] = calculate_distance(src, subsurface, self.cort)
                        vtx += 1
                    
                if not len(subsurface['subsurface_translated']) or not len(subsurface['data']):
                    # Get preprocessed timeseries within the given ROI and depth
                    self.logger.info('Extracting preprocessed timeseries for current ROI and cortical surface...')
                    self.cfm_output_config[aperture_type][subsurf_name]['subsurface_translated'], non_occ_vtx, flag_str = \
                        translate_indices(original_mask = self.occ_mask,
                                          new_mask = self.cfm_output_config[aperture_type][subsurf_name]['subsurface'],
                                          depth = self.cfm_output_config[aperture_type][subsurf_name]['depth'],
                                          target_surfs = self.target_surfs)
                    if flag_str is not None:
                        self.logger.info(flag_str)

                        # remove the non-occipital indices from the distance matrix:
                        self.cfm_output_config[aperture_type][subsurf_name]['dist'] = np.delete(self.cfm_output_config[aperture_type][subsurf_name]['dist'], non_occ_vtx, axis=0)
                        self.cfm_output_config[aperture_type][subsurf_name]['dist'] = np.delete(self.cfm_output_config[aperture_type][subsurf_name]['dist'], non_occ_vtx, axis=1)

                    self.cfm_output_config[aperture_type][subsurf_name]['data'] = self.cf_run_config[aperture_type]['preproc_data_concatenated_depths']
        
                    self.logger.info('Subsurface {} created.'.format(subsurf_name))

                    ## Save subsurfaces
                    self.logger.info('Saving subsurface...')
                    with open(self.cfm_config.output_data_dict_fn, 'wb') as pickle_file:
                        pickle.dump(self.cfm_output_config, pickle_file)
                else:
                    self.logger.info('Subsurface {} already exists.'.format(subsurf_name))

            self.logger.info('All subsurfaces for aperture type: {} created.'.format(aperture_type))