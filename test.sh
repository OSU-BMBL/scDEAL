#!/usr/bin/bash
#SBATCH --job-name ori
#SBATCH --account PCON0022
#SBATCH --time=00:10:00
#SBATCH --nodes=1 
#SBATCH --ntasks=1
#SBATCH --mem=5GB
#SBATCH --gpus-per-node=1


module load python/3.7-2019.10
module load cuda/10.1.168
source activate deal
drug='Cisplatin'
#dataname="GSE117872"
#edim="512,256"
#pdim="64,32"
#model='AE'
#dout=128
mod='new'
#python main.py --select_drug $drug --data_name $dataname --encoder_hdims $edim --preditpr_hdims $pdim --reduce_model $model --dim_au_out $dout
#GSE117872 HN137
python bulkmodel.py --drug 'CISPLATIN'  --dimreduce 'DAE' --encoder_h_dims "512,256" --predictor_h_dims "128,64" --bottleneck 32  --data_name 'GSE117872_HN137' --sampling 'SMOTE' --dropout 0.0 --printgene 'T' --mod $mod --lr 0.01
python scmodel.py --sc_data 'GSE117872_HN137' --dimreduce 'DAE' --drug 'CISPLATIN'  --bulk_h_dims "512,256" --bottleneck 32 --predictor_h_dims "128,64" --dropout 0.0 --printgene 'T' --mod $mod --sampling 'SMOTE' --lr 0.01
#rm '/users/PAS1475/anjunma/wxy/scDEAL/saved/models/'*

#GSE117872 HN120
python bulkmodel.py --drug 'CISPLATIN'  --dimreduce 'DAE' --encoder_h_dims "256,128" --predictor_h_dims "256,128" --bottleneck 512 --data_name 'GSE117872_HN120' --sampling 'downsampling' --dropout 0.1 --printgene 'T'  --mod $mod --lr 0.1
python scmodel.py --sc_data 'GSE117872_HN120' --dimreduce 'DAE' --drug 'CISPLATIN'  --bulk_h_dims "256,128" --bottleneck 512 --predictor_h_dims "256,128" --batch_id 'HN120' --dropout 0.1 --printgene 'T' --mod $mod --sampling 'downsampling' --lr 0.1
#rm '/users/PAS1475/anjunma/wxy/scDEAL/saved/models/'*

#GSE140440
python bulkmodel.py --drug 'DOCETAXEL' --dimreduce 'DAE' --encoder_h_dims "256,128" --predictor_h_dims "256,128" --bottleneck 512 --data_name 'GSE140440' --sampling 'no' --dropout 0.3 --printgene 'T' --mod $mod --lr 0.01
python scmodel.py --sc_data 'GSE140440' --dimreduce 'DAE' --drug 'DOCETAXEL'  --bulk_h_dims "256,128" --bottleneck 512 --predictor_h_dims "256,128" --dropout 0.3 --printgene 'T' --mod $mod --sampling 'no' --lr 0.01
#rm '/users/PAS1475/anjunma/wxy/scDEAL/saved/models/'*

#GSE149383
python bulkmodel.py --drug 'ERLOTINIB' --dimreduce 'DAE' --encoder_h_dims "512,256" --predictor_h_dims "256,128" --bottleneck 512  --data_name 'GSE149383' --sampling 'SMOTE' --dropout 0.1 --printgene 'T' --mod $mod --lr 0.1
python scmodel.py --sc_data 'GSE149383' --dimreduce 'DAE' --drug 'ERLOTINIB'  --bulk_h_dims "512,256" --bottleneck 512 --predictor_h_dims "256,128" --dropout 0.1 --printgene 'T' --mod $mod --sampling 'SMOTE' --lr 0.1
#rm '/users/PAS1475/anjunma/wxy/scDEAL/saved_new/models/'*

#GSE112274
python bulkmodel.py --drug 'GEFITINIB' --dimreduce 'DAE' --encoder_h_dims "512,256" --predictor_h_dims "256,128" --bottleneck 64 --data_name 'GSE112274' --sampling 'no' --dropout 0.1 --printgene 'T' --mod $mod --lr 0.1
python scmodel.py --sc_data 'GSE112274' --dimreduce 'DAE' --drug 'GEFITINIB'  --bulk_h_dims "512,256" --bottleneck 64 --predictor_h_dims "256,128" --dropout 0.1 --printgene 'T' --mod $mod --sampling 'no' --lr 0.1
#rm '/users/PAS1475/anjunma/wxy/scDEAL/saved/models/'*

#GSE110894
python bulkmodel.py --drug 'I.BET.762' --dimreduce 'DAE' --encoder_h_dims "512,256" --predictor_h_dims "256,128" --bottleneck 128 --data_name 'GSE110894' --sampling 'downsampling' --dropout 0.1 --printgene 'T' --mod $mod --lr 0.01
python scmodel.py --sc_data 'GSE110894' --dimreduce 'DAE' --drug 'I.BET.762'  --bulk_h_dims "512,256" --bottleneck 128 --predictor_h_dims "256,128" --dropout 0.1 --printgene 'T' --mod $mod --sampling 'downsampling' --lr 0.01
#rm '/users/PAS1475/anjunma/wxy/scDEAL/saved/models/'*



