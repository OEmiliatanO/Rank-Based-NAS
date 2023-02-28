# Genetic Based Neural Architecture Search with Hybrid Score Function

## Usage 

Create a conda environment using the env.yml file

```bash
conda env create -f env.yml
```

Activate the environment and follow the instructions to install

Install nasbench (see https://github.com/google-research/nasbench)

Download the NDS data from https://github.com/facebookresearch/nds and place the json files in naswot-codebase/nds_data/
Download the NASbench101 data (see https://github.com/google-research/nasbench)
Download the NASbench201 data (see https://github.com/D-X-Y/NAS-Bench-201)

Reproduce all of the results by running 

```bash
./scorehook.sh
```

The code is licensed under the MIT licence.

## Acknowledgement
- code based from NASWOT
- code based from NAS-bench-201

