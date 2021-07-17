# scDEAL
Deep Transfer Learning of Drug Sensitivity by Integrating Bulk and Single-cell RNA-seq data

## Libiries to run to model
Following envrionment or packages are required to run the programm:

- python 3.7.3
- torch 1.3.0
- sklearn 0.23.0
- imblearn 0.7.0
- scanpy 1.6.0

## Usage
Two main scripts to run the program are bulkmodel.py and scmodel.py
Run bulkmode.py first using the python commond line to train the source model.
For examples, run bulkmode.py with user defined prarmaters:

```
python bulkmodel.py --drug I-BET-762 -e saved/models/bulk_encoder_ae_256.pkl -p saved/models/bulk_predictor_  --dimreduce AE --encoder_h_dims "256,256" --predictor_h_dims "128,64" --bottleneck 256 

```

This step will train a drug sensitivity predictor for the drug 'I-BET-762.' The output model will be stroed in the directory "saved/models." The prefix of the model's file name will be 'bulk_predictor_ae_' and its full name will be dependent to paramters that users insert. In this case. The file name of the bulk model will be "bulk_predictor_AEI-BET-762.pkl". Then we can run: 

```
python scmodel.py --sc_data data/data.csv --pretrain saved/models/sc_encoder_ae.pkl -s saved/models/bulk_predictor_AEI-BET-762.pkl --dimreduce AE --sc_model_path saved/models/sc_predictor --drug I-BET-762  --bulk_h_dims "256,256" --bottleneck 256 --predictor_h_dims "128,64"
```
This step train the scDEAL model and generated predict the sensitivity of I-BET-762 of your input scRNA-Seq data. Remember that the dimention of the encoder and predictor should be identical (--bulk_h_dims "256,256" --bottleneck 256) in two steps. The output format of scDEAL are the [AnnData](https://anndata.readthedocs.io/en/latest/anndata.AnnData.html) object (.h5ad) applied by the scanpy package. The file will be stroed in the directory "saved\adata\data\". The prediction of sensitivity will be stored in adata.obs["sens_label"] (if you load your AnnDdata object named as adata) where 0 represents resistant and 1 represents sensitivty respectively.

For more detailed settings of two scripts, pleas refer to the documentation section.

## Documentation


* bulkmodel.py
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
* scmodel.py

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

# Software environment
The software is developed and tested in the environment:
```
Python 3.7.3 (default, Apr 24 2019, 15:29:51) [MSC v.1915 64 bit (AMD64)] :: Anaconda, Inc. on win32
```
