prf-workflow is a package that contains a full cortical surface-based population receptive field (pRF) modelling pipeline. Notably, the package can be used on high-resolution data for multi-surface pRF mapping. The package is equally well suited to standard single-surface pRF mapping.

As inputs, the package requires the preprocessed pRF run data (coregistered to the anatomical data), Freesurfer outputs and the stimulus apertures that were used for pRF stimulus presentation.

The package then carries out the following steps:
- Configuration [config.py]
    - input file names and locations
    - pRF fitting parameters
    - stimulus aperture parameters 
    - data cleaning preferences
- Image Processing [img_utils.py]
    - (for multi-surface pRF mapping) equivolumetric surface generation using [surface_tools](https://github.com/kwagstyl/surface_tools)
    - surface projection of the pRF runs
    - cleaning the pRF run data
        - detrending
        - standardizing
        - bandpass filtering  
        - removing confounds (e.g., )
- PRF mapping [prfpy_interface.py]
    *This is the main part of the package and acts as a wrapper to the [pRFpy](https://github.com/VU-Cog-Sci/prfpy/tree/main) package from the Spinoza Centre for Neuroimaging, Amsterdam*
    - creating a stimulus object for prfpy
    - fitting isotropic (Iso) two-dimensional Gaussian model to single or average surface
        *if analysis is being run on multiple cortical surfaces, it is possible to average the signals across cortical surfaces for an average fit, which can then be used to initialize the individual surface fits for greater robustness*
    - extracting final pRF parameters from fitting results and saving them as .mgh surface files for delineation purposes as well as into a dictionary for further processing:
        prf_params[aperture_type][param_name][depth]
    - (optional) fitting difference-of-Gaussians (DoG) model
    - (for multi-surface pRF mapping) fitting Iso 2D Gaussian or DoG model to each individual surface
    
To get started using these components in your own software, clone this repository, then run python installer.py.


If you notice any bugs or typos or have any suggestions, we would really value your input. Either send us a pull request or email us at maya [dot] jastrzebowska [at] gmail [dot] com.