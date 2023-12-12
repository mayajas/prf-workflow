# img_utils.py
from fsl.data.freesurfer import loadVertexDataFile
from nilearn import surface, signal
import nipype.interfaces.freesurfer as fs
import numpy as np
import os
from os.path import join as opj
import pickle
import subprocess
import sys

def is_running_on_slurm():
    return "SLURM_JOB_ID" in os.environ

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
            self.logger.info('Analysis will be run on wm surface.') # TODO: implement other single-surface options

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

    def __init__(self, project_config, dir_config, mri_config, logger):
        self.subject_id     = project_config.subject_id
        self.hemi           = project_config.hemi
        self.dir_config     = dir_config
        self.mri_config     = mri_config
        self.prf_run_config = mri_config.prf_run_config
        self.n_surfs        = project_config.n_surfs
        self.logger         = logger

        # surface project mean functionals
        source_file = self.mri_config.meanFunc_nii_fn
        if self.n_surfs > 1:
            projection_surface     = self.mri_config.equi_surf_fn_list[0]
        else:
            projection_surface     = 'white'
        out_file    = self.mri_config.meanFunc_mgh_fn
        if not os.path.exists(out_file):
            logger.info('Surface-projecting mean functional...')
            logger.info('Source file: {}'.format(source_file))
            logger.info('Surface: {}'.format(projection_surface))
            logger.info('Output file: {}'.format(out_file))
            if not os.path.exists(out_file):
                try:
                    logger.info('FS_dir: {}'.format(self.dir_config.FS_dir))
                    self._surface_project(self.dir_config.FS_dir,self.subject_id,self.hemi,source_file,projection_surface,out_file)
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
                        projection_surface = 'white'
                    out_file = config['mgh_fn_list'][run][depth]
                    if not os.path.exists(out_file):
                        logger.info('Source file: {}'.format(source_file))
                        logger.info('Surface: {}'.format(projection_surface))
                        logger.info('Output file: {}'.format(out_file))
                        try:
                            self._surface_project(self.dir_config.FS_dir,self.subject_id,self.hemi,source_file,projection_surface,out_file)
                            logger.info('Surface-projecting pRF mapping run {} of {} for {} aperture type and {} depth completed.'.format(run+1,config['n_runs'],aperture_type,depth+1))
                        except Exception as e:
                            logger.error(f"Error: {str(e)}")
                            logger.exception("Full exception traceback:")
                            sys.exit(1)
                    else:
                        logger.info('run {} of {} for depth {} already surface-projected.'.format(run+1,config['n_runs'],depth+1))
                        
    def _surface_project(self,FS_dir,subject_id,hemi,source_file,projection_surface,out_file):    
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
        sampler.inputs.surface = projection_surface
        sampler.inputs.out_file = out_file
        sampler.run()

class CleanInputData:
    """
    This class contains functions that are used to clean input data.
    """
    def __init__(self, project_config, prf_config, mri_config, data_clean_config, logger):
        self.n_surfs            = project_config.n_surfs
        self.occ_mask_fn        = mri_config.occ_mask_fn
        self.prf_run_config     = mri_config.prf_run_config
        self.y_coord_cutoff     = prf_config.y_coord_cutoff
        self.input_data_dict_fn = prf_config.input_data_dict_fn
        self.logger             = logger
        
        ## Load preprocessed surface meshes
        logger.info('Loading GM/WM surface meshes...')
        self.gm_mesh        = surface.load_surf_mesh(mri_config.gm_surf_fn) 
        self.wm_mesh        = surface.load_surf_mesh(mri_config.wm_surf_fn) 

        ## Load surface-projected data (mean functional and pRF runs across all apertures, runs, depths)
        logger.info('Loading surface-projected data...')
        self.meanFunc_mgh   = loadVertexDataFile(mri_config.meanFunc_mgh_fn)
        for aperture_type, config in self.prf_run_config.items():
            config['raw_data'] = {}  # Add a new key 'raw_data' for each aperture type
            for run in range(0,config['n_runs']):
                config['raw_data'][run] = {}
                for depth in range(0,self.n_surfs):
                    mgh_fn = config['mgh_fn_list'][run][depth]
                    config['raw_data'][run][depth] = loadVertexDataFile(mgh_fn)

        ## Make occipital mask
        self.occ_mask, self.n_vtx = self._make_occipital_mask()

        ## Clean input data
        # - apply occipital mask to constrain analysis to occipital pole
        # - detrend, standardize, and bandpass filter each functional pRF run
        # - average pRF runs
        self.detrend        = data_clean_config.detrend
        self.standardize    = data_clean_config.standardize
        self.low_pass       = data_clean_config.low_pass
        self.high_pass      = data_clean_config.high_pass     
        self.TR             = data_clean_config.TR
        self.filter         = data_clean_config.filter
        self.confounds      = data_clean_config.confounds

        mri_config.prf_run_config = self._clean_data()

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
    
    def _clean_data(self):
        
        if not os.path.exists(self.input_data_dict_fn):
            self.logger.info('Cleaning input data...')
            for aperture_type, config in self.prf_run_config.items():
                self.logger.info('Cleaning data for {} aperture type...'.format(aperture_type))
                config['filtered_data'] = {}  # Add a new key 'filtered_data' for each aperture type
                config['masked_data'] = {}  # Add a new key 'masked_data' for each aperture type
                config['preproc_data_per_depth'] = {}  # Add a new key 'preproc_data_per_depth' for each aperture type
                config['preproc_data_avg'] = {}  # Add a new key 'preproc_data_avg' for each aperture type
                for run in range(0,config['n_runs']):
                    self.logger.info('Run {} of {}...'.format(run+1,config['n_runs']))
                    config['masked_data'][run] = {}
                    config['filtered_data'][run] = {}
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
                        
                # Average over runs
                self.logger.info('Averaging over runs...')
                config['preproc_data_per_depth'] = [0] * self.n_surfs
                for run in config['filtered_data']:
                    for depth in range(0,self.n_surfs):
                        config['preproc_data_per_depth'][depth] += config['filtered_data'][run][depth]

                for depth in range(0,self.n_surfs):
                    config['preproc_data_per_depth'][depth] = config['preproc_data_per_depth'][depth].T
                    config['preproc_data_per_depth'][depth] /= config['n_runs']

                # Average over depths
                self.logger.info('Averaging over depths...')
                config['preproc_data_avg'] = sum(config['preproc_data_per_depth']) / self.n_surfs

            ## Save cleaned data
            self.logger.info('Saving cleaned data...')
            with open(self.input_data_dict_fn, 'wb') as pickle_file:
                pickle.dump(self.prf_run_config, pickle_file)
        else:
            self.logger.info('Cleaned data already exists at {}'.format(self.input_data_dict_fn))
            self.logger.info('Loading cleaned data...')
            with open(self.input_data_dict_fn, 'rb') as pickle_file:
                self.prf_run_config = pickle.load(pickle_file)

            # Check if the cleaned data matches the dimensions of the occipital mask
            for aperture_type, config in self.prf_run_config.items():
                for run in range(0,config['n_runs']):
                    for depth in range(0,self.n_surfs):
                        if config['masked_data'][run][depth].shape[1] != self.occ_mask.shape[0]:
                            self.logger.error('Cleaned data does not match the dimensions of the occipital mask.')
                            self.logger.error('It is suggested to delete the cleaned data dictionary and the occipital mask and rerun the analysis.')
                            sys.exit(1)

            # Check if the cleaned data dictionary has all the needed keys ('filtered_data', 'masked_data', 'preproc_data_per_depth', 'preproc_data_avg')
            for aperture_type, config in self.prf_run_config.items():
                for key in ['filtered_data', 'masked_data', 'preproc_data_per_depth', 'preproc_data_avg']:
                    if key not in config:
                        self.logger.error('Cleaned data dictionary does not have the key: {}'.format(key))
                        self.logger.error('Cleaned data dictionary must have the following keys: {}'.format(['filtered_data', 'masked_data', 'preproc_data_per_depth', 'preproc_data_avg']))
                        self.logger.error('Cleaned data dictionary has the following keys: {}'.format(list(config.keys())))
                        self.logger.error('It is suggested to delete the cleaned data dictionary and rerun the analysis.')
                        sys.exit(1)

        return self.prf_run_config
        