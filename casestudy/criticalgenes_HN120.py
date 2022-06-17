import pandas as pd
import scanpy as sc
import glob
import argparse


parser = argparse.ArgumentParser()
# data 
parser.add_argument('--sc_data', type=str, default="GSE117872_HN120",help='Accession id for testing data, only support pre-built data.')
# miss
parser.add_argument('--input_path', '-i',  type=str, default='save/logs/transfer_',help='Path of training log')
parser.add_argument('--output_path','-o', type=str, default=None,help='Samping method of training data for the bulk model traning. \
                    Can be upsampling, downsampling, or SMOTE. default: None')
parser.add_argument('--fix_source', type=int, default=0,help='Fix the bulk level model. Default: 0')
parser.add_argument('--bulk', type=str, default='integrate',help='Selection of the bulk database.integrate:both dataset. old: GDSC. new: CCLE. Default: integrate')


args, unknown = parser.parse_known_args()

DATA = args.sc_data
h5adpath="/users/PAS1475/anjunma/wxy/scDEAL/save/adata/*"+DATA+"*h5ad"
sc_gradpath="/users/PAS1475/anjunma/wxy/scDEAL/ori_result/*"+DATA+"sc_gradient.txt"
sc_genepath="/users/PAS1475/anjunma/wxy/scDEAL/ori_result/*"+DATA+"sc_gene.csv"
CGpath=args.output_path


def get_ct(data): 
    if(data=="GSE110894"):
        return("Sample name")
    elif(data=="GSE117872_HN120"):
        return("cell_color")
    elif(data=="GSE117872_HN137"):
        return("cell_color")
    else:
        return("leiden")
    

PCT_EXP = 25
ORDER = 'log_fc'
# Change it to your path that sabe h5ad for HN120 critical gene gradient
f_cg = glob.glob(sc_gradpath)
df_cg = pd.read_csv(f_cg[0],sep=" ",header=None)
f_gn = glob.glob(sc_genepath)
df_gene = pd.read_csv(f_gn[0],index_col=0)

df_cg.columns = df_gene.iloc[:,0].values
adata_fname = glob.glob(h5adpath)
adata = sc.read_h5ad(adata_fname[1])
adata.obs.head()
df_cg.index = adata.obs.index

df_cg = abs(df_cg)


igadata= sc.AnnData(df_cg)
igadata.var.index = adata.var.index
igadata.obs.index = adata.obs.index
sc.pp.filter_cells(igadata, min_genes=200)
sc.pp.filter_genes(igadata, min_cells=3)
sc.pl.highest_expr_genes(igadata, n_top=20, )
sc.pp.normalize_total(igadata)
sc.pp.log1p(igadata)
sc.pp.highly_variable_genes(igadata, min_mean=0.0125, max_mean=3, min_disp=0.5)
igadata.obs['sens_label'] = adata.obs['sens_label'].astype('category')
rg_result = sc.tl.rank_genes_groups(igadata, 'sens_label',method='wilcoxon',pts=True)


map_ref_group = {
    1: ["HN120P",'HN120PCR'],
    2: ['HN120P',"HN120M"],
    3: ['HN120PCR','HN120MCR'],
    4: ['HN120P','HN120MCR'],
    5: ['HN120M','HN120MCR'],
    6: ['HN120M','HN120PCR']
}


for i in range(1,7):
    subset_igdata = igadata[igadata.obs['cell_color'].isin(map_ref_group[i])]
    rg_result = sc.tl.rank_genes_groups(subset_igdata, 'cell_color', groups=[map_ref_group[i][0]], reference=map_ref_group[i][1], method='wilcoxon',pts =True)
    #sc.pl.rank_genes_groups(subset_igdata, groups=[map_ref_group[i]], n_genes=20)
    df_name = pd.DataFrame(subset_igdata.uns['rank_genes_groups']['names'])#.to_csv("critical_genes_experiment_"+str(i)+".csv")
    df_lgfc = pd.DataFrame(subset_igdata.uns['rank_genes_groups']['logfoldchanges'])#.to_csv("critical_logfc_experiment_"+str(i)+".csv")
    df_pval = pd.DataFrame(subset_igdata.uns['rank_genes_groups']['pvals_adj'])#.to_csv("critical_genes_experiment_"+str(i)+".csv")
    df_pts = pd.DataFrame(subset_igdata.uns['rank_genes_groups']['pts'])#.to_csv("critical_logfc_experiment_"+str(i)+".csv")

    for c in df_name.columns:
        tmp_df = pd.DataFrame({"gene_name":df_name[c],"log_fc":df_lgfc[c],'pvals_adj':df_pval[c],'pts':df_pts.loc[df_name[c].values,c].values})
        tmp_df.to_csv(CGpath+ map_ref_group[i][0]+"_"+map_ref_group[i][1]+"2.csv")

    sc.pl.stacked_violin(subset_igdata, pd.DataFrame(subset_igdata.uns['rank_genes_groups']['names']).head(10).T.values.ravel(), groupby='cell_color',\
                        rotation=180,save="critical_genes_violint_"+str(i)+".svg");
                        
gene = pd.read_csv(CGpath+"HN120P_HN120PCR.csv",)
gene = gene[gene['log_fc'].abs()>0.1]
gene = gene[gene['pvals_adj']<=0.05]
gene = gene[gene['pts']>0.2]
gene=gene.sort_values("log_fc",ascending=True)
res = list(gene[gene['log_fc']<0]['gene_name'])
gene=gene.sort_values("log_fc",ascending=False)
sen=list(gene[gene['log_fc']>0]['gene_name'])
