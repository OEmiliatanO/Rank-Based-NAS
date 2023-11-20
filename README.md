# Rank Based Neural Architecture Search

## Usage 

Create a conda environment using the env.yml file

```bash
conda env create -f env.yml
```

Activate the environment and follow the instructions to install

Install nasbench (see https://github.com/google-research/nasbench)

Download the NAS-Bench-101 data (see https://github.com/google-research/nasbench)
Download the NATS-Bench-TSS data (see https://github.com/D-X-Y/NATS-Bench)
Download the NATS-Bench-SSS data (see https://github.com/D-X-Y/NATS-Bench)

Reproduce all of the results by running 

```bash
./search.sh
```

## Acknowledgement
- code based on NAS-bench-101
- code based on NAS-bench-TSS
- code based on NATS-bench-SSS
- code based on NASWOT
- code based on TNASSE
- code based on FreeREA
