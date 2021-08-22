# scDEAL documentation
Deep Transfer Learning of Drug Sensitivity by Integrating Bulk and Single-cell RNA-seq data

## System requirements

Following packages are required to run the program:

- conda 4.8.4
- python 3.7.3
- torch 1.3.0
- sklearn 0.23.0
- imblearn 0.7.0
- scanpy 1.6.0

This software is developed and tested in the following software environment:
```
Python 3.7.3 (default, Apr 24 2019, 15:29:51) [MSC v.1915 64 bit (AMD64)] :: Anaconda, Inc. on win32
```

This software is developed and tested on the hardware envrionments of:
- CPU: HexaCore Intel Core i7-9750H, 4100 MHz (41 x 100)
- RAM: 32GB
- GPU: nVIDIA GeForce GTX 1660 Ti with Max-Q Design

No non-standard hardware are required for the software.

## Installation guide

The software is a stand alone python scirpt package. It can be download and installed with this github repository:

```
git clone https://github.com/OSU-BMBL/scDEAL.git
```
Installation time is depened on the network speed of user.

## Data preparation

Please create two directories 'data' and 'data/GSE110894' and store the data download from the [data](https://bmbl.bmi.osumc.edu/downloadFiles/scdeal/) link inside. The organization of the directory should be similar as follows:

```
scDEAL
│   README.md
│   bulkmodel.py  
│   scmodel.py
|   ...
└───data
│   │   GDSC1_expression.csv
│   │   GDSC2_expression.csv
│   │   GDSC2_expression.csv
│   │   GDSC1_label_345drugs_binary.csv
│   │   GDSC2_label_192drugs_binary.csv
│   │   GDSC1_label_9drugs_binary.csv
│   │   ...
│   └───GSE110894
│       │   GSE110894.csv
│       │   GSE110894_CellInfo.xlsx
│       │   ...
└───saved
│   │   
|   └───logs
│   │    ...
|   └───figures
│   │    ...
|   └───models
│   │    ...
│   └───adata
│   |    │
│   |    └───data
│   │    ...   
└───DaNN
│   │    ...   
└───scanpypip
│   │    ...  
└───figures
│   │    ...
```

## Directory contents

Folders in our package will store the corresponding contents:

- root: python scripts to run the program and README.md
- data: datasets required for the learning
- saved/logs: log and error files that recording running status. 
- saved/figures & figures: figures generated through the run. 
- saved/models: models triained through the run. 
- saved/adata: results AnnData outputs.
- DaNN: python scripts describe the model.
- scanpypip: python scripts of utilities.

## Usage
Two main scripts to run the program are bulkmodel.py and scmodel.py
Run bulkmode.py first using the python command line to train the source model.

For example, run bulkmode.py with user-defined parameters:

```
python bulkmodel.py --drug I-BET-762 -e saved/models/bulk_encoder_ae_256.pkl -p saved/models/bulk_predictor_  --dimreduce AE --encoder_h_dims "256,256" --predictor_h_dims "128,64" --bottleneck 256 

```

This step takes the expression profile of bulk RNA-Seq and the drug response annotations as input. Iw will train a drug sensitivity predictor for the drug 'I-BET-762.' The output model will be stored in the directory "saved/models." The prefix of the model's file name will be 'bulk_predictor_ae_' and its full name will be dependent on parameters that users insert. In this case. The file name of the bulk model will be "bulk_predictor_AEI-BET-762.pkl". For all available drug names, please refer to the columns names of files: GDSC*_label_*drugs_binary.csv. 

For the transfer learning, we provide a built-in testing case of acute myeloid leukemia cells [Bell et al.](https://doi.org/10.1038/s41467-019-10652-9) accessed from Gene Expression Omnibus (GEO) accession [GSE110894](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE110894). The training time of the cases are:  


```
python scmodel.py --sc_data GSE110894 --pretrain saved/models/sc_encoder_ae.pkl -s saved/models/bulk_predictor_AEI-BET-762.pkl --dimreduce AE --sc_model_path saved/models/sc_predictor --drug I-BET-762  --bulk_h_dims "256,256" --bottleneck 256 --predictor_h_dims "128,64"
```
This step trains the scDEAL model and generated predict the sensitivity of I-BET-762 of the input scRNA-Seq data from GSE110984. Remember that the dimension of the encoder and predictor should be identical (--bulk_h_dims "256,256" --bottleneck 256) in two steps. 

For your input count matrix, you can replace the --sc_data option with your data path as follows:

```
python scmodel.py --sc_data [*Your own data path*] ...
```

The training time of the test case including bulk-level and single-cell-level training on the testing computer was 4 minutes.

The expected output format of scDEAL is the [AnnData](https://anndata.readthedocs.io/en/latest/anndata.AnnData.html) object (.h5ad) applied by the scanpy package. The file will be stored in the directory "saved\adata\data\". The prediction of sensitivity will be stored in adata.obs["sens_label"] (if you load your AnnDdata object named as adata) where 0 represents resistance and 1 represents sensitivity respectively. Further analysis for the output can be processed by the package [Scanpy](https://scanpy.readthedocs.io/en/stable/). The object can be loaded in to python through the function: [scanpy.read_h5ad](https://scanpy.readthedocs.io/en/latest/generated/scanpy.read_h5ad.html#scanpy-read-h5ad). 

The expected output format of a successful run show includes:

```
scDEAL
|   ...
└───data
│   │   ...
└───saved
│   └───adata
│   |    │
│   |    └───data
│   │        GSE110894[a timestamp].h5ad   
│   │    ...   
|   └───models
│   │    bulk_endoder_ae_256.pkl
│   │    bulk_predictor_AEIBET-762.pkl
│   │    sc_predictor_DaNN.pkl
│   │    ...
```

For more detailed settings of the two scripts, please refer to the documentation section.

## Documentation


* Command: python bulkmodel.py
```
usage: bulkmodel.py [-h] [--data DATA] [--label LABEL] [--result RESULT]
                    [--drug DRUG] [--missing_value MISSING_VALUE]
                    [--test_size TEST_SIZE] [--valid_size VALID_SIZE]
                    [--var_genes_disp VAR_GENES_DISP] [--sampling SAMPLING]
                    [--PCA_dim PCA_DIM] [--bulk_encoder BULK_ENCODER]
                    [--pretrain PRETRAIN] [--lr LR] [--epochs EPOCHS]
                    [--batch_size BATCH_SIZE] [--bottleneck BOTTLENECK]
                    [--dimreduce DIMREDUCE]
                    [--freeze_pretrain FREEZE_PRETRAIN]
                    [--encoder_h_dims ENCODER_H_DIMS]
                    [--predictor_h_dims PREDICTOR_H_DIMS]
                    [--VAErepram VAEREPRAM] [--bulk_model BULK_MODEL]
                    [--log LOG] [--load_source_model LOAD_SOURCE_MODEL]

optional arguments:
  -h, --help            show this help message and exit
  --data DATA           Path of the bulk RNA-Seq expression profile
  --label LABEL         Path of the processed bulk RNA-Seq drug screening
                        annotation
  --result RESULT       Path of the training result report files
  --drug DRUG           Name of the selected drug, should be a column name in
                        the input file of --label
  --missing_value MISSING_VALUE
                        The value filled in the missing entry in the drug
                        screening annotation, default: 1
  --test_size TEST_SIZE
                        Size of the test set for the bulk model traning,
                        default: 0.2
  --valid_size VALID_SIZE
                        Size of the validation set for the bulk model traning,
                        default: 0.2
  --var_genes_disp VAR_GENES_DISP
                        Dispersion of highly variable genes selection when
                        pre-processing the data. If None, all genes will be
                        selected .default: None
  --sampling SAMPLING   Samping method of training data for the bulk model
                        traning. Can be upsampling, downsampling, or SMOTE.
                        default: None
  --PCA_dim PCA_DIM     Number of components of PCA reduction before training.
                        If 0, no PCA will be performed. Default: 0
  --bulk_encoder BULK_ENCODER, -e BULK_ENCODER
                        Path of the pre-trained encoder in the bulk level
  --pretrain PRETRAIN   Whether to perform pre-training of the encoder. 0: do
                        not pretraing, 1: pretrain. Default: 0
  --lr LR               Learning rate of model training. Default: 1e-2
  --epochs EPOCHS       Number of epoches training. Default: 500
  --batch_size BATCH_SIZE
                        Number of batch size when training. Default: 200
  --bottleneck BOTTLENECK
                        Size of the bottleneck layer of the model. Default: 32
  --dimreduce DIMREDUCE
                        Encoder model type. Can be AE or VAE. Default: AE
  --freeze_pretrain FREEZE_PRETRAIN
                        Fix the prarmeters in the pretrained model. 0: do not
                        freeze, 1: freeze. Default: 0
  --encoder_h_dims ENCODER_H_DIMS
                        Shape of the encoder. Each number represent the number
                        of neuron in a layer. Layers are seperated by a comma.
                        Default: 512,256
  --predictor_h_dims PREDICTOR_H_DIMS
                        Shape of the predictor. Each number represent the
                        number of neuron in a layer. Layers are seperated by a
                        comma. Default: 16,8
  --VAErepram VAEREPRAM
  --bulk_model BULK_MODEL, -p BULK_MODEL
                        Path of the trained prediction model in the bulk level
  --log LOG, -l LOG     Path of training log
  --load_source_model LOAD_SOURCE_MODEL
                        Load a trained bulk level or not. 0: do not load, 1:
                        load. Default: 0
```
* Command: python scmodel.py

```
usage: scmodel.py [-h] [--bulk_data BULK_DATA] [--label LABEL]
                  [--sc_data SC_DATA] [--drug DRUG]
                  [--missing_value MISSING_VALUE] [--test_size TEST_SIZE]
                  [--valid_size VALID_SIZE] [--var_genes_disp VAR_GENES_DISP]
                  [--min_n_genes MIN_N_GENES] [--max_n_genes MAX_N_GENES]
                  [--min_g MIN_G] [--min_c MIN_C]
                  [--percent_mito PERCENT_MITO] [--cluster_res CLUSTER_RES]
                  [--mmd_weight MMD_WEIGHT] [--mmd_GAMMA MMD_GAMMA]
                  [--bulk_model_path BULK_MODEL_PATH]
                  [--sc_model_path SC_MODEL_PATH] [--pretrain PRETRAIN]
                  [--lr LR] [--epochs EPOCHS] [--batch_size BATCH_SIZE]
                  [--bottleneck BOTTLENECK] [--dimreduce DIMREDUCE]
                  [--freeze_pretrain FREEZE_PRETRAIN]
                  [--bulk_h_dims BULK_H_DIMS] [--sc_h_dims SC_H_DIMS]
                  [--predictor_h_dims PREDICTOR_H_DIMS]
                  [--VAErepram VAEREPRAM] [--batch_id BATCH_ID]
                  [--load_sc_model LOAD_SC_MODEL]
                  [--logging_file LOGGING_FILE]

optional arguments:
  -h, --help            show this help message and exit
  --bulk_data BULK_DATA
                        Path of the bulk RNA-Seq expression profile
  --label LABEL         Path of the processed bulk RNA-Seq drug screening
                        annotation
  --sc_data SC_DATA     Accession id for testing data, only support pre-built
                        data.
  --drug DRUG           Name of the selected drug, should be a column name in
                        the input file of --label
  --missing_value MISSING_VALUE
                        The value filled in the missing entry in the drug
                        screening annotation, default: 1
  --test_size TEST_SIZE
                        Size of the test set for the bulk model traning,
                        default: 0.2
  --valid_size VALID_SIZE
                        Size of the validation set for the bulk model traning,
                        default: 0.2
  --var_genes_disp VAR_GENES_DISP
                        Dispersion of highly variable genes selection when
                        pre-processing the data. If None, all genes will be
                        selected .default: None
  --min_n_genes MIN_N_GENES
                        Minimum number of genes for a cell that have UMI
                        counts >1 for filtering propose, default: 0
  --max_n_genes MAX_N_GENES
                        Maximum number of genes for a cell that have UMI
                        counts >1 for filtering propose, default: 20000
  --min_g MIN_G         Minimum number of genes for a cell >1 for filtering
                        propose, default: 200
  --min_c MIN_C         Minimum number of cell that each gene express for
                        filtering propose, default: 3
  --percent_mito PERCENT_MITO
                        Percentage of expreesion level of moticondrial genes
                        of a cell for filtering propose, default: 100
  --cluster_res CLUSTER_RES
                        Resolution of Leiden clustering of scRNA-Seq data,
                        default: 0.3
  --mmd_weight MMD_WEIGHT
                        Weight of the MMD loss of the transfer learning,
                        default: 0.25
  --mmd_GAMMA MMD_GAMMA
                        Gamma parameter in the kernel of the MMD loss of the
                        transfer learning, default: 1000
  --bulk_model_path BULK_MODEL_PATH, -s BULK_MODEL_PATH
                        Path of the trained predictor in the bulk level
  --sc_model_path SC_MODEL_PATH, -p SC_MODEL_PATH
                        Path (prefix) of the trained predictor in the single
                        cell level
  --pretrain PRETRAIN   Path of the pre-trained encoder in the single-cell
                        level
  --lr LR               Learning rate of model training. Default: 1e-2
  --epochs EPOCHS       Number of epoches training. Default: 500
  --batch_size BATCH_SIZE
                        Number of batch size when training. Default: 200
  --bottleneck BOTTLENECK
                        Size of the bottleneck layer of the model. Default: 32
  --dimreduce DIMREDUCE
                        Encoder model type. Can be AE or VAE. Default: AE
  --freeze_pretrain FREEZE_PRETRAIN
                        Fix the prarmeters in the pretrained model. 0: do not
                        freeze, 1: freeze. Default: 0
  --bulk_h_dims BULK_H_DIMS
                        Shape of the source encoder. Each number represent the
                        number of neuron in a layer. Layers are seperated by a
                        comma. Default: 512,256
  --sc_h_dims SC_H_DIMS
                        Shape of the encoder. Each number represent the number
                        of neuron in a layer. Layers are seperated by a comma.
                        Default: 512,256
  --predictor_h_dims PREDICTOR_H_DIMS
                        Shape of the predictor. Each number represent the
                        number of neuron in a layer. Layers are seperated by a
                        comma. Default: 16,8
  --VAErepram VAEREPRAM
  --batch_id BATCH_ID   Batch id only for testing
  --load_sc_model LOAD_SC_MODEL
                        Load a trained model or not. 0: do not load, 1: load.
                        Default: 0
  --logging_file LOGGING_FILE, -l LOGGING_FILE
                        Path of training log
```

