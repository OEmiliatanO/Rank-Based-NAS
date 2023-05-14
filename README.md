# Rank Based Neural Architecture Search with Hybrid Score Functions

## Usage 

Create a conda environment using the env.yml file

```bash
conda env create -f env.yml
```

Activate the environment and follow the instructions to install

Install nasbench (see https://github.com/google-research/nasbench)

Download the NASbench101 data (see https://github.com/google-research/nasbench)
Download the NASbench201 data (see https://github.com/D-X-Y/NAS-Bench-201)
Download the NATSbenchSSS data

Reproduce all of the results by running 

```bash
./search.sh
```

## Acknowledgement
- code based on NAS-bench-101
- code based on NAS-bench-201
- code based on NATS-bench-SSS
- code based on NASWOT
- code based on TNASSE
- code based on FreeREA
