# scDEAL documentation
Deep Transfer Learning of Drug Sensitivity by Integrating Bulk and Single-cell RNA-seq data

## News 2023/03/19
Multiple package versions changed and therefore the results shown in the article cannot be fully replicated, but we provide the full result information (embedding, sensitive score and so on) for all of our data, stored in the adata format. These results can be downloaded from [here](https://portland-my.sharepoint.com/:u:/g/personal/junyichen8-c_my_cityu_edu_hk/EYru-LaQC1tHlFZSnf1RA_cBjXwIafy-iDsajEWjh8xcjA?e=2sE61e).   
Now, we are re-tuning parameters based on the current version environment and we will show the results on GitHub. 

This update provides all the shell files to help you repeat all the results.
1. Added the function of loading checkpoint weights to the model.
2. Added the conda environment applied to produce the result.
3. Reorganized and uploaded the resources need for the code.

## News 2022/12/04
1. Add trained adata.h5ad objects for 6 data examples.

## Previous News 
1. Added the function to use clustering labels to help transfer learning. Details are listed in the usage section.
2. Migrate the source of testing data from the FTP to OneDrive.

## Installation 

### Retrieve code from GitHub
The software is a stand-alone python script package. The home directory of scDEAL can be cloned from the GitHub repository:

```
# Clone from Github
git clone https://github.com/OSU-BMBL/scDEAL.git
# Go into the directory
cd scDEAL
```

It’s recommended to install the scDEAL under **Linux** and install the provided conda environment through the conda pack [Click here to download scdeal.tar.gz](https://portland-my.sharepoint.com/:u:/g/personal/junyichen8-c_my_cityu_edu_hk/EaOYJmIATDdFoI5wqcDJiVsBV8wtnved8LDP7pwqf7T5jQ?e=UHqeya). It’s recommended to install in your root conda environment - the conda pack command will then be available in all sub-environments as well.

[Click here to download scdeal.tar.gz](https://portland-my.sharepoint.com/:u:/g/personal/junyichen8-c_my_cityu_edu_hk/EaOYJmIATDdFoI5wqcDJiVsBV8wtnved8LDP7pwqf7T5jQ?e=UHqeya) 


### Install with conda:
conda-pack is available from Anaconda as well as from conda-forge:
```
conda install conda-pack
conda install -c conda-forge conda-pack
```
### Install from PyPI:
While conda-pack requires an existing conda install, it can also be installed from PyPI:
```
pip install conda-pack
```

## Load the scDEALenv environment
conda-pack is primarily a command line tool. Full CLI docs can be found here.
One common use case is packing an environment on one machine to distribute to other machines that may not have conda/python installed.
Place the downloaded scdeal.tar.gz into your scDEAL folder. Import and activate the environment of your target machine:
```
# Unpack environment into directory `scDEALenv`
mkdir -p scDEALenv
tar -xzf scDEAL.tar.gz -C scDEALenv
# Activate the environment. This adds `scDEALenv/bin` to your path
source scDEALenv/bin/activate
```

## Data Preparation
### Data download
After setting up the home directory, you need to download other resources required for the run. Please create and download the zip format dataset from the [scDEAL.zip](https://portland-my.sharepoint.com/:u:/r/personal/junyichen8-c_my_cityu_edu_hk/Documents/scDEAL/0319/scDEAL.zip?csf=1&web=1&e=Bbul9m) link inside:

[Click here to download scDEAL.zip](https://portland-my.sharepoint.com/:u:/r/personal/junyichen8-c_my_cityu_edu_hk/Documents/scDEAL/0319/scDEAL.zip?csf=1&web=1&e=Bbul9m) 

The file "scDEAL.zip" includes all the datasets we have tested. Please extract the zip file and place the sub-directory "data" in the root directory of the "scDEAL" folder. 
|               |     Author             |     Drug         |     GEO access    |     Cells    |     Species           |     Cancer type                        |
|---------------|------------------------|------------------|-------------------|--------------|-----------------------|----------------------------------------|
|     Data 1&2  |     Sharma, et al.     |     Cisplatin    |     GSE117872     |     548      |     Homo   sapiens    |     Oral   squamous cell carcinomas    |
|     Data 3    |     Kong, et al.       |     Gefitinib    |     GSE112274     |     507      |     Homo   sapiens    |     Lung   cancer                      |
|     Data 4    |     Schnepp, et al.    |     Docetaxel    |     GSE140440     |     324      |     Homo   sapiens    |     Prostate   Cancer                  |
|     Data 5    |     Aissa, et al.      |     Erlotinib    |     GSE149383     |     1496     |     Homo sapiens      |     Lung cancer                        |
|     Data 6    |     Bell, et al.       |     I-BET-762    |     GSE110894     |     1419     |     Mus   musculus    |     Acute   myeloid leukemia           |

"scDEAL.zip" also includes model checkpoints in the "save" directory. Try to extract the scDEAL.zip.
```
# Unpack scDEAL.zip into directory `scDEAL`
unzip scDEAL.zip
# View folder
ls -a
#ls results:
#bulkmodel.py  DaNN  LICENSE    README.md    save       scDEALenv   trainers.py    utils.py
#casestudy     data  models.py  sampling.py  scanpypip  scmodel.py  trajectory.py
```
All resources in the home directory of scDEAL should look as follows:

```
scDEAL
└───scDEALenv
|   ...
│   README.md
│   bulkmodel.py  
│   scmodel.py
|   ...
└───data
│   │   ALL_expression.csv
│   │   ALL_label_binary_wf.csv
│   └───GSE110894
│   └───GSE112274
│   └───GSE117872
│   └───GSE140440
│   └───GSE149383
│   |   ...
└───save
|   └───logs
|   └───figures
|   └───models
│   │   └───bulk_encoder
│   │   └───bulk_pre
│   │   └───sc_encoder
│   │   └───sc_pre
│   └───adata
│   │    ...   
└───DaNN
└───scanpypip
│   │    ...  
```

### Directory contents
Folders in our package will store the corresponding contents:

- root: python scripts to run the program and README.md
- data: datasets required for the learning
- save/logs: log and error files that record running status. 
- save/figures & figures: figures generated through the run. 
- save/models: models trained through the run. 
- save/adata: results from AnnData outputs.
- DaNN: python scripts describe the model.
- scanpypip: python scripts of utilities.

## Demo
### Pretrained checkpoints
For the scRNA-Seq prediction task, we provide pre-trained checkpoints for the models stored in save/models 
The naming rules of the checkpoint are as follows:

bulk dataset+"_data_"+scRNA-Seq dataset+"_drug_"+drug+"_bottle_"+bottle+"_edim_"+encoder dimensions+"_pdim_"+predictor dimensions+"_model_"+encoder model+"_dropout_"+dropout+"_gene_"+show critical genes+"_lr_"+learning rate+"_mod_"+model version+"_sam_"+sampling method(+"_DANN.pkl"only in the final single cell model)    

An example can be:

integrate_data_GSE110894_drug_I.BET.762_bottle_512_edim_256,128_pdim_128,64_model_DAE_dropout_0.1_gene_F_lr_0.5_mod_new_sam_upsampling_DaNN.pkl

Usage:
For resuming training, you can use the --checkpoint option of scmodel.py and bulkmodel.py.
For example, run scmodel.py with checkpoints to get the single-cell level prediction results:

```
source scDEALenv/bin/activate
python scmodel.py --sc_data "GSE110894" --dimreduce "DAE" --drug "I.BET.762" --bulk_h_dims "256,128" --bottleneck 512 --predictor_h_dims "128,64" --dropout 0.1 --printgene "F" -mod "new" --lr 0.5 --sampling "upsampling" --printgene "F" -mod "new" --checkpoint "save/sc_pre/integrate_data_GSE110894_drug_I.BET.762_bottle_512_edim_256,128_pdim_128,64_model_DAE_dropout_0.1_gene_F_lr_0.5_mod_new_sam_upsampling_DaNN.pkl"
```
This step is a built-in testing case of acute myeloid leukemia cells [Bell](https://doi.org/10.1038/s41467-019-10652-9) et al.](https://doi.org/10.1038/s41467-019-10652-9) accessed from Gene Expression Omnibus (GEO) accession [GSE110894](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE110894). This step calls the scDEAL model and predicts the sensitivity of I.BET.762 of the input scRNA-Seq data from GSE110984. The file name of the single cell model is "save/sc_pre/integrate_data_GSE110894_drug_I.BET.762_bottle_512_edim_256,128_pdim_128,64_model_DAE_dropout_0.1_gene_F_lr_0.5_mod_new_sam_upsampling_DaNN.pkl". The we also provide the checkpoint from the bulk level and run bulkmodel.py with checkpoints and then the scmodel.py to get the single-cell level prediction results

```
source scDEALenv/bin/activate
python bulkmodel.py --drug "I.BET.762" --dimreduce "DAE" --encoder_h_dims "256,128" --predictor_h_dims "128,64" --bottleneck 512 --data_name "GSE110894" --sampling "upsampling" --dropout 0.1 --lr 0.5 --printgene "F" -mod "new" --checkpoint "save/bulk_pre/integrate_data_GSE110894_drug_I.BET.762_bottle_512_edim_256,128_pdim_128,64_model_DAE_dropout_0.1_gene_F_lr_0.5_mod_new_sam_upsampling"

python scmodel.py --sc_data "GSE110894" --dimreduce "DAE" --drug "I.BET.762" --bulk_h_dims "256,128" --bottleneck 512 --predictor_h_dims "128,64" --dropout 0.1 --printgene "F" -mod "new" --lr 0.5 --sampling "upsampling" --printgene "F" -mod "new" --checkpoint "save/sc_pre/integrate_data_GSE110894_drug_I.BET.762_bottle_512_edim_256,128_pdim_128,64_model_DAE_dropout_0.1_gene_F_lr_0.5_mod_new_sam_upsampling_DaNN.pkl"
```
Remember that the dimension of the encoder and predictor should be identical (--encoder_h_dims(bulk_h_dims) "256,128", --predictor_h_dims "128,64", --bottleneck 256) in two steps. This step takes the expression profile of bulk RNA-Seq and the drug response annotations as input. It loads a drug sensitivity predictor for the drug "I.BET.762." The output model is stored in the directory "save/models." In this case. The file name of the bulk model is "save/bulk_pre/integrate_data_GSE110894_drug_I.BET.762_bottle_512_edim_256,128_pdim_128,64_model_DAE_dropout_0.1_gene_F_lr_0.5_mod_new_sam_upsampling". 

### Train from scratch
Suggested parameters for our selected datasets are as follows

|           data |      drug | bottleneck | encoder dimensions | predictor dimensions | encoder model | dropout | learning rate |   sampling |
|---------------:|----------:|-----------:|-------------------:|---------------------:|--------------:|--------:|--------------:|-----------:|
|      GSE110894 | I.BET.762 |        512 |            256,128 |               128,64 |           DAE |     0.1 |           0.5 | upsampling |
|      GSE112274 | GEFITINIB |        256 |            512,256 |              256,128 |           DAE |     0.1 |           0.5 |         no |
| GSE117872HN120 | CISPLATIN |        512 |            256,128 |               128,64 |           DAE |     0.3 |          0.01 |      SMOTE |
| GSE117872HN137 | CISPLATIN |         32 |            512,256 |              256,128 |           DAE |     0.3 |          0.01 | upsampling |
|      GSE140440 | DOCETAXEL |        512 |            256,128 |              256,128 |           DAE |     0.1 |          0.01 | upsampling |
|      GSE149383 | ERLOTINIB |         64 |            512,256 |              256,128 |           DAE |     0.3 |          0.01 | upsampling |

We can also train bulkmode.py and scmodel.py from scratch with user-defined parameters by setting --checkpoint "False":
```
source scDEALenv/bin/activate
python bulkmodel.py --drug "I.BET.762" --dimreduce "DAE" --encoder_h_dims "256,128" --predictor_h_dims "128,64" --bottleneck 512 --data_name "GSE110894" --sampling "upsampling" --dropout 0.1 --lr 0.5 --printgene "F" -mod "new" --checkpoint "False"
python scmodel.py --sc_data "GSE110894" --dimreduce "DAE" --drug "I.BET.762" --bulk_h_dims "256,128" --bottleneck 512 --predictor_h_dims "128,64" --dropout 0.1 --printgene "F" -mod "new" --lr 0.5 --sampling "upsampling" --printgene "F" -mod "new" --checkpoint "False"
```
Remember that the dimension of the encoder and predictor should be identical (--encoder_h_dims(bulk_h_dims) "256,128", --predictor_h_dims "128,64", --bottleneck 256) in two steps. This step train the scDEAL model and predicts the sensitivity of I.BET.762 of the input scRNA-Seq data from GSE110984.  

### Expected run time for the demo
The training time of the test case including bulk-level and single-cell-level training on the testing computer was 20 minutes on the pytorch cpu version.

### Expected output
The expected output format of scDEAL is the [AnnData](https://anndata.readthedocs.io/en/latest/anndata.AnnData.html) object (.h5ad) applied by the scanpy package. The file will be stored in the directory "scDEAL/adata/data/". The prediction of sensitivity will be stored in adata.obs["sens_label"] (if you load your AnnDdata object named as adata) where 0 represents resistance and 1 represents sensitivity respectively. Further analysis for the output can be processed by the package [Scanpy](https://scanpy.readthedocs.io/en/stable/). The object can be loaded into python through the function: [scanpy.read_h5ad](https://scanpy.readthedocs.io/en/latest/generated/scanpy.read_h5ad.html#scanpy-read-h5ad). 

The expected output format of a successful run show includes:

```
scDEAL
|   ...
└───data
│   │   ...
└───save
│   └───adata
│   |    │
│   |    └───data
│   │        GSE110894[a timestamp].h5ad   
│   │    ...   
|   └───models
│   │    save/bulk_encoder
│   │    save/bulk_pre
│   │    save/sc_encoder
│   │    save/sc_pre
│   │    ...
```
### Run the software on your data
For your input count matrix, you can replace the --sc_data option with your data path as follows:

```
source scDEALenv/bin/activate
python bulkmodel.py --drug [*Your selected drug*] --data [*Your own bulk level expression*] --label [*Your own bulk level drug resistance table*] ...
python scmodel.py --sc_data [*Your own data path*] ...
```

Formats for your drug resistance table and your bulk level expression should be identical to the files in the demo:
- [ALL_label_binary_wf.csv](https://portland-my.sharepoint.com/:u:/r/personal/junyichen8-c_my_cityu_edu_hk/Documents/scDEAL/0319/scDEAL.zip?csf=1&web=1&e=Bbul9m)
- [ALL_expression.csv](https://portland-my.sharepoint.com/:u:/r/personal/junyichen8-c_my_cityu_edu_hk/Documents/scDEAL/0319/scDEAL.zip?csf=1&web=1&e=Bbul9m)

For more detailed parameter settings of the two scripts, please refer to the documentation section.

## * Appendix
### * Appendix A: case studies
The folder named "casestudy" contains Jupyter notebook templates of selected case studies in the paper. You can follow the introduction within each notebook to create analysis results for scDEAL. For example, run bulkmode.py with user-defined parameters:

```
python bulkmodel.py --drug [*Your selected drug*] --data [*Your own bulk level expression*] --label [*Your own bulk level drug resistance table*] ...  --printgene "T"

python scmodel.py --sc_data [*Your own data path*] ... --printgene "T"
```
- [casestudy/analysis_criticalgenes.ipynb](casestudy/analysis_criticalgenes.ipynb): critical gene identification by integrated gradient matrix; 
- [casestudy/analysis_tarined_anndata.ipynb](casestudy/analysis_tarined_anndata.ipynb): umap, gene score, and regression plot of single-cell level prediction.


### * Appendix B: trained .h5ad files
Trained results as scanpy h5ad objects for the example datasets are provided as follows:
- [adata.zip](https://portland-my.sharepoint.com/:u:/g/personal/junyichen8-c_my_cityu_edu_hk/EYru-LaQC1tHlFZSnf1RA_cBjXwIafy-iDsajEWjh8xcjA?e=2sE61e)


## Documentation
* Command: python bulkmodel.py
```
usage: bulkmodel.py [-h] [--data DATA] [--label LABEL] [--result RESULT] [--drug DRUG] [--missing_value MISSING_VALUE] [--test_size TEST_SIZE] [--valid_size VALID_SIZE]
                    [--var_genes_disp VAR_GENES_DISP] [--sampling SAMPLING] [--PCA_dim PCA_DIM] [--device DEVICE] [--bulk_encoder BULK_ENCODER] [--pretrain PRETRAIN] [--lr LR] [--epochs EPOCHS]
                    [--batch_size BATCH_SIZE] [--bottleneck BOTTLENECK] [--dimreduce DIMREDUCE] [--freeze_pretrain FREEZE_PRETRAIN] [--encoder_h_dims ENCODER_H_DIMS]
                    [--predictor_h_dims PREDICTOR_H_DIMS] [--VAErepram VAEREPRAM] [--data_name DATA_NAME] [--checkpoint CHECKPOINT] [--bulk_model BULK_MODEL] [--log LOG]
                    [--load_source_model LOAD_SOURCE_MODEL] [--mod MOD] [--printgene PRINTGENE] [--dropout DROPOUT] [--bulk BULK]

optional arguments:
  -h, --help            show this help message and exit
  --data DATA           Path of the bulk RNA-Seq expression profile
  --label LABEL         Path of the processed bulk RNA-Seq drug screening annotation
  --result RESULT       Path of the training result report files
  --drug DRUG           Name of the selected drug, should be a column name in the input file of --label
  --missing_value MISSING_VALUE
                        The value filled in the missing entry in the drug screening annotation, default: 1
  --test_size TEST_SIZE
                        Size of the test set for the bulk model traning, default: 0.2
  --valid_size VALID_SIZE
                        Size of the validation set for the bulk model traning, default: 0.2
  --var_genes_disp VAR_GENES_DISP
                        Dispersion of highly variable genes selection when pre-processing the data. If None, all genes will be selected .default: None
  --sampling SAMPLING   Samping method of training data for the bulk model traning. Can be upsampling, downsampling, or SMOTE. default: no
  --PCA_dim PCA_DIM     Number of components of PCA reduction before training. If 0, no PCA will be performed. Default: 0
  --device DEVICE       Device to train the model. Can be cpu or gpu. Deafult: cpu
  --bulk_encoder BULK_ENCODER, -e BULK_ENCODER
                        Path of the pre-trained encoder in the bulk level
  --pretrain PRETRAIN   Whether to perform pre-training of the encoder,str. False: do not pretraing, True: pretrain. Default: True
  --lr LR               Learning rate of model training. Default: 1e-2
  --epochs EPOCHS       Number of epoches training. Default: 500
  --batch_size BATCH_SIZE
                        Number of batch size when training. Default: 200
  --bottleneck BOTTLENECK
                        Size of the bottleneck layer of the model. Default: 32
  --dimreduce DIMREDUCE
                        Encoder model type. Can be AE or VAE. Default: AE
  --freeze_pretrain FREEZE_PRETRAIN
                        Fix the prarmeters in the pretrained model. 0: do not freeze, 1: freeze. Default: 0
  --encoder_h_dims ENCODER_H_DIMS
                        Shape of the encoder. Each number represent the number of neuron in a layer. Layers are seperated by a comma. Default: 512,256
  --predictor_h_dims PREDICTOR_H_DIMS
                        Shape of the predictor. Each number represent the number of neuron in a layer. Layers are seperated by a comma. Default: 16,8
  --VAErepram VAEREPRAM
  --data_name DATA_NAME
                        Accession id for testing data, only support pre-built data.
  --checkpoint CHECKPOINT
                        Load weight from checkpoint files, can be True,False, or file path. Checkpoint files can be paraName1_para1_paraName2_para2... Default: True
  --bulk_model BULK_MODEL, -p BULK_MODEL
                        Path of the trained prediction model in the bulk level
  --log LOG, -l LOG     Path of training log
  --load_source_model LOAD_SOURCE_MODEL
                        Load a trained bulk level or not. 0: do not load, 1: load. Default: 0
  --mod MOD             Embed the cell type label to regularized the training: new: add cell type info, ori: do not add cell type info. Default: new
  --printgene PRINTGENE
                        Print the cirtical gene list: T: print. Default: T
  --dropout DROPOUT     Dropout of neural network. Default: 0.3
  --bulk BULK           Selection of the bulk database.integrate:both dataset. old: GDSC. new: CCLE. Default: integrate
  ```
* Command: python scmodel.py

```
usage: scmodel.py [-h] [--bulk_data BULK_DATA] [--label LABEL] [--sc_data SC_DATA] [--drug DRUG] [--missing_value MISSING_VALUE] [--test_size TEST_SIZE] [--valid_size VALID_SIZE]
                  [--var_genes_disp VAR_GENES_DISP] [--min_n_genes MIN_N_GENES] [--max_n_genes MAX_N_GENES] [--min_g MIN_G] [--min_c MIN_C] [--percent_mito PERCENT_MITO]
                  [--cluster_res CLUSTER_RES] [--mmd_weight MMD_WEIGHT] [--mmd_GAMMA MMD_GAMMA] [--device DEVICE] [--bulk_model_path BULK_MODEL_PATH] [--sc_model_path SC_MODEL_PATH]
                  [--sc_encoder_path SC_ENCODER_PATH] [--checkpoint CHECKPOINT] [--lr LR] [--epochs EPOCHS] [--batch_size BATCH_SIZE] [--bottleneck BOTTLENECK] [--dimreduce DIMREDUCE]
                  [--freeze_pretrain FREEZE_PRETRAIN] [--bulk_h_dims BULK_H_DIMS] [--sc_h_dims SC_H_DIMS] [--predictor_h_dims PREDICTOR_H_DIMS] [--VAErepram VAEREPRAM] [--batch_id BATCH_ID]
                  [--load_sc_model LOAD_SC_MODEL] [--mod MOD] [--printgene PRINTGENE] [--dropout DROPOUT] [--logging_file LOGGING_FILE] [--sampling SAMPLING] [--fix_source FIX_SOURCE]
                  [--bulk BULK]

optional arguments:
  -h, --help            show this help message and exit
  --bulk_data BULK_DATA
                        Path of the bulk RNA-Seq expression profile
  --label LABEL         Path of the processed bulk RNA-Seq drug screening annotation
  --sc_data SC_DATA     Accession id for testing data, only support pre-built data.
  --drug DRUG           Name of the selected drug, should be a column name in the input file of --label
  --missing_value MISSING_VALUE
                        The value filled in the missing entry in the drug screening annotation, default: 1
  --test_size TEST_SIZE
                        Size of the test set for the bulk model traning, default: 0.2
  --valid_size VALID_SIZE
                        Size of the validation set for the bulk model traning, default: 0.2
  --var_genes_disp VAR_GENES_DISP
                        Dispersion of highly variable genes selection when pre-processing the data. If None, all genes will be selected .default: None
  --min_n_genes MIN_N_GENES
                        Minimum number of genes for a cell that have UMI counts >1 for filtering propose, default: 0
  --max_n_genes MAX_N_GENES
                        Maximum number of genes for a cell that have UMI counts >1 for filtering propose, default: 20000
  --min_g MIN_G         Minimum number of genes for a cell >1 for filtering propose, default: 200
  --min_c MIN_C         Minimum number of cell that each gene express for filtering propose, default: 3
  --percent_mito PERCENT_MITO
                        Percentage of expreesion level of moticondrial genes of a cell for filtering propose, default: 100
  --cluster_res CLUSTER_RES
                        Resolution of Leiden clustering of scRNA-Seq data, default: 0.3
  --mmd_weight MMD_WEIGHT
                        Weight of the MMD loss of the transfer learning, default: 0.25
  --mmd_GAMMA MMD_GAMMA
                        Gamma parameter in the kernel of the MMD loss of the transfer learning, default: 1000
  --device DEVICE       Device to train the model. Can be cpu or gpu. Deafult: cpu
  --bulk_model_path BULK_MODEL_PATH, -s BULK_MODEL_PATH
                        Path of the trained predictor in the bulk level
  --sc_model_path SC_MODEL_PATH, -p SC_MODEL_PATH
                        Path (prefix) of the trained predictor in the single cell level
  --sc_encoder_path SC_ENCODER_PATH
                        Path of the pre-trained encoder in the single-cell level
  --checkpoint CHECKPOINT
                        Load weight from checkpoint files, can be True,False, or a file path. Checkpoint files can be paraName1_para1_paraName2_para2... Default: True
  --lr LR               Learning rate of model training. Default: 1e-2
  --epochs EPOCHS       Number of epoches training. Default: 500
  --batch_size BATCH_SIZE
                        Number of batch size when training. Default: 200
  --bottleneck BOTTLENECK
                        Size of the bottleneck layer of the model. Default: 32
  --dimreduce DIMREDUCE
                        Encoder model type. Can be AE or VAE. Default: AE
  --freeze_pretrain FREEZE_PRETRAIN
                        Fix the prarmeters in the pretrained model. 0: do not freeze, 1: freeze. Default: 0
  --bulk_h_dims BULK_H_DIMS
                        Shape of the source encoder. Each number represent the number of neuron in a layer. Layers are seperated by a comma. Default: 512,256
  --sc_h_dims SC_H_DIMS
                        Shape of the encoder. Each number represent the number of neuron in a layer. Layers are seperated by a comma. Default: 512,256
  --predictor_h_dims PREDICTOR_H_DIMS
                        Shape of the predictor. Each number represent the number of neuron in a layer. Layers are seperated by a comma. Default: 16,8
  --VAErepram VAEREPRAM
  --batch_id BATCH_ID   Batch id only for testing
  --load_sc_model LOAD_SC_MODEL
                        Load a trained model or not. 0: do not load, 1: load. Default: 0
  --mod MOD             Embed the cell type label to regularized the training: new: add cell type info, ori: do not add cell type info. Default: new
  --printgene PRINTGENE
                        Print the cirtical gene list: T: print. Default: T
  --dropout DROPOUT     Dropout of neural network. Default: 0.3
  --logging_file LOGGING_FILE, -l LOGGING_FILE
                        Path of training log
  --sampling SAMPLING   Samping method of training data for the bulk model traning. Can be no, upsampling, downsampling, or SMOTE. default: no
  --fix_source FIX_SOURCE
                        Fix the bulk level model. Default: 0
  --bulk BULK           Selection of the bulk database.integrate:both dataset. old: GDSC. new: CCLE. Default: integrate
```

