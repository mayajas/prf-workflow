import logging
import os.path.join as opj

def setup_logger(project_config):
    """
    This function sets up a logger for the project.

    Args:
        project_config (ProjectConfig object): Contains project configuration parameters.

    Returns:
        logger (Logger object): Logger
    """

    # Get variables from project_config
    subject_id  = project_config.subject_id
    hemi        = project_config.hemi
    logger_dir  = project_config.logger_dir
    
    # Create a logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)  # Set the logging level to INFO or your preferred level

    # Create a file handler and set the format
    # TODO: change log file name to include model_name
    log_file = f"prf_{subject_id}_{hemi}.log"
    file_handler = logging.FileHandler(opj(logger_dir,log_file), mode='w')  # Use 'w' mode to overwrite the file
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    return logger