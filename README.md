A proposed solution to the Entity Resultion Category of the 2nd DI2KG Workshop Challenge (Monitor Data, Domain knowledge approach). This is a Scientific Team  Project with a team size of 6.

# Comments:
A solution highly tailored to the domain, with the core strategies of: information propagation for model detection, carefully-tuned brand and model extraction (with domain-specific choices).

# Program assumptions (other than requirements):
### The domain-specific choices we made are limited to: 
    1. cleaning of site-specific texts for better TF-IDF results
    2. non-exhaustive brand (attribute) keywords, extracted by looking at some examples of the data 
    3. brand names that were extracted with a bit of a human-in-the-loop process (where we saw the brand names emerging and collected alternative names),
    4. a large amount of rules for brand cleaning, resulting from data understanding (this is the less general aspect of our solution)... the amount of hand-crafted configurations really show the amount of time the team spent exploring and understanding the data, 
    5. Rules for extracting the models
    6. Cleaning of false-positive model names.

We consider the hard-coded rules to deter from our generality. However, we include them since they are crucial for finding a straight-forward solution with the limited resources chosen.

# Individual Contribution
    1. Data cleaning for each website provided in the data.
    2. Creating text embeddings using BERT.
    3. Creating rules based in brand name specific to monitors as a product and non-product category (If brand is a monitor brand then Product else Non_product).
    4. Using Machine Learning model like K-Neareast neighbour to classify into products and non-products based on created rules.
    
    
# How to Run the code?
### Run the python file with this command : 
python3 SchemaMatching.py


# Installed Python Libraries:
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


# Note : 
### This is an ongoing project and any changes done will be pushed in this repository. This repository is a part of the code submission on the DI2KG Workshop Challenge (Monitor Data, Domain knowledge approach).
