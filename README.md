# EFtranslocator
Predict various features relevant to plant host-cell translocation and sub-cellular localisation, for a given protein FASTA input


# Install apt/pip dependencies:
```
conda create --name EFtranslocator python=3.10
conda activate EFtranslocator
apt-get update
apt-get install -y \
    build-essential=12.9ubuntu3 \
    emboss=6.6.0+dfsg-11ubuntu1 \
    libgraphviz-dev=2.42.2-6ubuntu0.1 \
    libsvm-tools=3.24+ds-6 \
    ncbi-blast+=2.12.0+ds-3build1 \
    openjdk-11-jdk=11.0.31+11-1ubuntu1~22.04.2 \
    python2=2.7.18-3 \
    python3-dev=3.10.6-1~22.04.1 \
    python3-pil=9.0.1-1ubuntu0.4 \
    python3-pygraphviz=1.7-3build1

pip install \
    wheel==0.47.0 \
    weka==2.0.2 \
    python_weka_wrapper3[plots,graphs]==0.3.3 \
    dbcan==5.2.9 \
    UpSetPlot==0.9.0 \
    hmmer==3.4.0.2 \
    biopython==1.85 \
    pandas==2.2.3 \
    numpy==2.0.2 \
    tensorflow==2.20.0 \
    pybiolib==1.4.151 \
    fair-esm==2.0.0 \
    plicat_model==0.1.0 \
    pygam==0.12.0 \
    scikit-learn==1.6.1 \
    openbabel==3.2.0
```

# Install 3rd-party bioinformatics tools
copy archive files to EFtranslocator/bin folder:
  ```
  SignalP 6.0 (software and license) https://services.healthtech.dtu.dk/services/SignalP-6.0/ 
  TargetP
  WoLFPSORT 
  DeepLoc2
  MultiLoc2
  ApoplastP 
  LOCALIZER
  Diamond
  RiPPMiner
  ```

Automatically installed:
  ```
  HMMer
  DeepTMHMM
  PLiCat
  AIUPred
  dbCAN
  ```

Additional data derived from:
  ```
  CPPSite2
  Pfam
  ```
