A proposed solution to the Entity Resultion Category of the 2nd DI2KG Workshop Challenge (Monitor Data, Domain knowledge approach). This is a Scientific Team  Project with a team size of 6.

**Comments:

A solution highly tailored to the domain, with the core strategies of: information propagation for model detection, carefully-tuned brand and model extraction (with domain-specific choices). 

************************************

**Program assumptions (other than requirements):

1) We assume 2013_monitor_specs to be in the same path. 


The domain-specific choices we made are limited to: 
a) cleaning of site-specific texts for better TF-IDF results, 
b) non-exhaustive brand (attribute) keywords, extracted by looking at some examples of the data 
c) brand names that were extracted with a bit of a human-in-the-loop process (where we saw the brand names emerging and collected alternative names),
d) a large amount of rules for brand cleaning, resulting from data understanding (this is the less general aspect of our solution)... the amount of hand-crafted configurations really show the amount of time the team spent exploring and understanding the data, 
e) Rules for extracting the models\nf) Cleaning of false-positive model names.

We consider the hard-coded rules to deter from our generality. However, we include them since they are crucial for finding a straight-forward solution with the limited resources chosen.

To the best of our knowledge, any recent version of scikit-learn and numpy would be compatible with our submitted solution.

**Installed Python Libraries:

    ca-certificates-2020.6.20  |       hecda079_0         145 KB  conda-forge
    certifi-2020.6.20          |   py38h32f6830_0         151 KB  conda-forge
    joblib-0.16.0              |             py_0         203 KB  conda-forge
    ld_impl_linux-64-2.34      |       h53a641e_5         616 KB  conda-forge
    libblas-3.8.0              |      17_openblas          11 KB  conda-forge
    libcblas-3.8.0             |      17_openblas          11 KB  conda-forge
    liblapack-3.8.0            |      17_openblas          11 KB  conda-forge
    libopenblas-0.3.10         |       h5ec1e0e_0         7.8 MB  conda-forge
    numpy-1.18.5               |   py38h8854b6b_0         5.2 MB  conda-forge
    python-3.8.3               |cpython_he5300dc_0        71.0 MB  conda-forge
    scikit-learn-0.23.1        |   py38h3a94b23_0         7.0 MB  conda-forge
    scipy-1.5.0                |   py38h18bccfc_0        18.7 MB  conda-forge
    setuptools-49.1.0          |   py38h32f6830_0         911 KB  conda-forge
    sqlite-3.32.3              |       hcee41ef_0         2.0 MB  conda-forge
    threadpoolctl-2.1.0        |     pyh5ca1d4c_0          15 KB  conda-forge

**How to Run the code?

python3 SchemaMatching.py

**Note : This is an ongoing project and any changes done will be pushed in this repository.
