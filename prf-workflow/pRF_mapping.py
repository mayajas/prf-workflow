#!~/anaconda3/envs/prf-workflow/bin/python
##/usr/bin/env python
import argparse
import sys

from logger import setup_logger
from config import ProjectConfig, DirConfig, MriConfig, StimApertureConfig, PrfMappingConfig, DataCleanConfig, CfModelingConfig
from img_utils import EquivolumetricSurfaces, SurfaceProject, CleanInputData, CreateSubsurfaces
from prfpy_interface import PrfpyStimulus, PrfFitting, CfStimulus, CfModeling

def main(config_file,sub_idx,hem_idx):
    """
    Run population receptive field (pRF) mapping on a single subject's single hemisphere.

    Args:
        config_file (str): Path to the configuration file, which contains the image processing and pRF analysis parameters.
        sub_idx (int): Index of the subject to analyze.
        hem_idx (int): Index of the hemisphere to run the analysis on (0 for 'lh', 1 for 'rh').

    Returns:
        None

    """
    project_config  = ProjectConfig(config_file, sub_idx, hem_idx)
    n_procs         = project_config.n_procs

    ## Set up logger
    logger = setup_logger(project_config)
    logger.info('Number of cores: ' + str(n_procs))
    
    try:
        ###########################################################################################
        ### Setup
        ## Set data directories 
        dir_config = DirConfig(config_file, project_config, logger)

        ## Get pRF output filenames
        prf_config = PrfMappingConfig(config_file, dir_config, project_config, logger)
        
        if not project_config.do_cf_modeling:
            ## Get input MRI data
            mri_config = MriConfig(config_file, project_config, dir_config, prf_config, logger)

        else:
            ## Get CF mapping info (if applicable)
            cfm_config = CfModelingConfig(config_file, project_config, dir_config, prf_config, logger)

            ## Get input MRI data
            mri_config = MriConfig(config_file, project_config, dir_config, prf_config, logger, cfm_config)


        ## Get aperture info
        StimApertureConfig(dir_config, mri_config, logger)

        ## Get data cleaning info
        data_clean_config = DataCleanConfig(config_file, mri_config)


        logger.info('Configuration ready.')

        ###########################################################################################
        ### Input data preprocessing
        ## Generate equivolumetric surfaces
        EquivolumetricSurfaces(project_config, dir_config, mri_config, logger)

        if not project_config.do_cf_modeling:
            ## Surface-project functional data (mean functional and pRF runs)
            SurfaceProject(project_config, dir_config, mri_config, logger)

            ## Clean input data
            CleanInputData(project_config, prf_config, mri_config, data_clean_config, logger)
        else:
            ## Surface-project functional data (mean functional, pRF runs and CF runs)
            SurfaceProject(project_config, dir_config, mri_config, logger, cfm_config)

            ## Clean input data
            CleanInputData(project_config, prf_config, mri_config, data_clean_config, logger, cfm_config)

            ## Generate subsurfaces (source ROI for CF modeling)
            CreateSubsurfaces(mri_config, cfm_config, logger)

        ###########################################################################################
        ### PRF mapping
        ## Create pRF stimulus object
        PrfpyStimulus(dir_config, mri_config,prf_config, logger)

        ## Fit pRF model
        PrfFitting(dir_config,mri_config,prf_config,project_config,logger)

        logger.info('')
        logger.info('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        logger.info('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        logger.info("pRF analysis completed successfully!")
        logger.info('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        logger.info('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        logger.info('')

        ###########################################################################################
        ### CF modeling
        if project_config.do_cf_modeling:
            ## Create CF stimulus objects
            CfStimulus(mri_config, cfm_config, logger)

            ## Fit CF model
            CfModeling(mri_config, cfm_config, logger)

            logger.info('')
            logger.info('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            logger.info('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            logger.info("CFM analysis completed successfully!")
            logger.info('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            logger.info('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            logger.info('')

        
    except FileNotFoundError as e:
        logger.error(f"Error: {str(e)}")
        logger.exception("Full exception traceback:")
        sys.exit(1)  # Exit with a non-zero code to indicate error
    except Exception as e:
        logger.error(f"Unknown error occurred: {str(e)}")
        logger.exception("Full exception traceback:")
        sys.exit(1)  # Exit with a non-zero code to indicate error

if __name__ == "__main__":
    
    ### Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run population receptive field (& optionally connective field) model fitting on data projected to cortical surfaces.")
    parser.add_argument("config_file", type=str, help="Path to the configuration file, which contains the image processing and pRF analysis parameters.")
    parser.add_argument("sub_idx", type=int, help="Index of the subject to analyze.")
    parser.add_argument("hem_idx", type=int, choices=[0,1], help="Hemisphere to run the analysis on. This is the index of the current hemisphere from the hemisphere list: ['lh','rh']")

    args = parser.parse_args()

    main(args.config_file, args.sub_idx, args.hem_idx)