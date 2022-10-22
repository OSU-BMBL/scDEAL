import logging
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from captum.attr import IntegratedGradients
from pandas import read_excel
from scipy.stats import mannwhitneyu
from sklearn.metrics import precision_recall_curve, roc_curve
import scanpypip.utils as ut

def highly_variable_genes(data, 
    layer=None, n_top_genes=None, 
    min_disp=0.5, max_disp=np.inf, min_mean=0.0125, max_mean=3, 
    span=0.3, n_bins=20, flavor='seurat', subset=False, inplace=True, batch_key=None, PCA_graph=False, PCA_dim = 50, k = 10, n_pcs=40):

    adata = sc.AnnData(data)

    adata.var_names_make_unique()  # this is unnecessary if using `var_names='gene_ids'` in `sc.read_10x_mtx`
    adata.obs_names_make_unique()


    if n_top_genes!=None:
        sc.pp.highly_variable_genes(adata,layer=layer,n_top_genes=n_top_genes,
        span=span, n_bins=n_bins, flavor='seurat_v3', subset=subset, inplace=inplace, batch_key=batch_key)

    else: 
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata,
        layer=layer,n_top_genes=n_top_genes,
        min_disp=min_disp, max_disp=max_disp, min_mean=min_mean, max_mean=max_mean, 
        span=span, n_bins=n_bins, flavor=flavor, subset=subset, inplace=inplace, batch_key=batch_key)

    if PCA_graph == True:
        sc.tl.pca(adata,n_comps=PCA_dim)
        X_pca = adata.obsm["X_pca"]
        sc.pp.neighbors(adata, n_neighbors=k, n_pcs=n_pcs)

        return adata.var.highly_variable,adata,X_pca


    return adata.var.highly_variable,adata

def save_arguments(args,now):
    args_strings =re.sub("\'|\"|Namespace|\(|\)","",str(args)).split(sep=', ')
    args_dict = dict()
    for item in args_strings:
        items = item.split(sep='=')
        args_dict[items[0]] = items[1]

    args_df = pd.DataFrame(args_dict,index=[now]).T
    args_df.to_csv("saved/logs/arguments_" +now + '.csv')

    return args_df

def plot_label_hist(Y,save=None):

    # the histogram of the data
    n, bins, patches = plt.hist(Y, 50, density=True, facecolor='g', alpha=0.75)

    plt.xlabel('Y values')
    plt.ylabel('Probability')
    plt.title('Histogram of target')
    # plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
    # plt.xlim(40, 160)
    # plt.ylim(0, 0.03)
    # plt.grid(True)
    if save == None:
        plt.show()
    else:
        plt.savefig(save)

# plot no skill and model roc curves
def plot_roc_curve(test_y,naive_probs,model_probs,title="",path="figures/roc_curve.pdf"):

    # plot naive skill roc curve
    fpr, tpr, _ = roc_curve(test_y, naive_probs)
    plt.plot(fpr, tpr, linestyle='--', label='Random')
    # plot model roc curve
    fpr, tpr, _ = roc_curve(test_y, model_probs)
    plt.plot(fpr, tpr, marker='.', label='Predition')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    plt.title(title)

    # show the plot
    if path == None:
        plt.show()
    else:
        plt.savefig(path)
    plt.close() 

# plot no skill and model precision-recall curves
def plot_pr_curve(test_y,model_probs,selected_label = 1,title="",path="figures/prc_curve.pdf"):
    # calculate the no skill line as the proportion of the positive class
    no_skill = len(test_y[test_y==selected_label]) / len(test_y)
    # plot the no skill precision-recall curve
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='Random')
    # plot model precision-recall curve
    precision, recall, _ = precision_recall_curve(test_y, model_probs)
    plt.plot(recall, precision, marker='.', label='Predition')
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # show the legend
    plt.legend()
    plt.title(title)

    # show the plot
    if path == None:
        plt.show()
    else:
        plt.savefig(path)
    plt.close() 


def specific_process(adata,dataname="",**kargs):
    if dataname =="GSE117872":
        select_origin = kargs['select_origin']
        adata = process_117872(adata,select_origin=select_origin)
    elif dataname == "GSE122843":
        adata = process_122843(adata)
    elif dataname == "GSE110894":
        adata = process_110894(adata)
    elif dataname == "GSE112274":
        adata = process_112274(adata)
    elif dataname == "GSE108383":
        adata = process_108383(adata)
    elif dataname == "GSE140440":
        adata = process_140440(adata)
    elif dataname == "GSE129730":
        adata = process_129730(adata)
    elif dataname == "GSE149383":
        adata = process_149383(adata)
    return adata
    
def process_108383(adata,**kargs):
    obs_names = adata.obs.index
    annotation_dict = {}
    for section in [0,1,2,3,4]:
        svals = [index.split("_")[section] for index in obs_names]
        annotation_dict["name_section_"+str(section+1)] = svals
    df_annotation=pd.DataFrame(annotation_dict,index=obs_names)
    adata.obs=df_annotation
    # adata.obs['name_section_3'].replace("par", "sensitive", inplace=True)
    # adata.obs['name_section_3'].replace("br", "resistant", inplace=True)
    # adata.obs['sensitive']=adata.obs['name_section_3']

    sensitive = [int(row.find("br")==-1) for row in adata.obs.loc[:,"name_section_3"]]
    sens_ = ['Resistant' if (row.find("br")!=-1) else 'Sensitive' for row in adata.obs.loc[:,"name_section_3"]]
    #adata.obs.loc[adata.obs.cluster=="Holiday","cluster"] = "Sensitive"
    adata.obs['sensitive'] = sensitive
    adata.obs['sensitivity'] = sens_

    # Cluster de score
    pval = 0.05
    n_genes = 50
    if "pval_thres" in kargs:
        pval=kargs['pval_thres']
    if "num_de" in kargs:
        n_genes = kargs['num_de']
    adata = de_score(adata=adata,clustername="sensitivity",pval=pval,n=n_genes)
    return adata

def process_117872(adata,**kargs):

    annotation = pd.read_csv('data/GSE117872/GSE117872_good_Data_cellinfo.txt',sep="\t",index_col="groups")
    for item in annotation.columns:
        #adata.obs[str(item)] = annotation.loc[:,item].convert_dtypes('category').values
        adata.obs[str(item)] = annotation.loc[:,item].astype("category")

    if "select_origin" in kargs:
        origin = kargs['select_origin']
        if origin!="all":
            selected=adata.obs['origin']==origin
            selected=selected.to_numpy('bool')
            adata = adata[selected, :]
    
    sensitive = [int(row.find("Resistant")==-1) for row in adata.obs.loc[:,"cluster"]]
    sens_ = ['Resistant' if (row.find("Resistant")!=-1) else 'Sensitive' for row in adata.obs.loc[:,"cluster"]]
    #adata.obs.loc[adata.obs.cluster=="Holiday","cluster"] = "Sensitive"
    adata.obs['sensitive'] = sensitive
    adata.obs['sensitivity'] = sens_

    # Cluster de score
    pval = 0.05
    n_genes = 50
    if "pval_thres" in kargs:
        pval=kargs['pval_thres']
    if "num_de" in kargs:
        n_genes = kargs['num_de']
    adata = de_score(adata=adata,clustername="sensitivity",pval=pval,n=n_genes)
    return adata

def process_122843(adata,**kargs):
    # Data specific preprocessing of cell info
    file_name = 'data/GSE122843/GSE122843_CellInfo.xlsx' # change it to the name of your excel file
    df_cellinfo = read_excel(file_name,header=2)
    df_cellinfo = df_cellinfo.fillna(method='pad')

    # Dictionary of DMSO between cell info and expression matrix
    match_dict={'DMSO':'DMSO (D7)',
            "DMSOw8":'DMSO (D56)',
            "IBET400":"400nM IBET",
           "IBET600":"600nM IBET",
           "IBET800":"800nM IBET",
           "IBETI1000":"1000nM IBET",
           "IBET1000w8":"1000nM IBET (D56)"}
    inv_match_dict = {v: k for k, v in match_dict.items()}

    index = [inv_match_dict[sn]+'_' for sn in df_cellinfo.loc[:,'Sample Name']]

    # Creat index in the count matrix style
    inversindex = index+df_cellinfo.loc[:,'Well Position']
    inversindex.name = 'Index'
    df_cellinfo.index = inversindex

    # Inner join of the obs adata information
    obs_merge = pd.merge(adata.obs,df_cellinfo,left_index=True,right_index=True,how='left')

    # Replace obs
    adata.obs = obs_merge
    
    return adata
def process_110894(adata,**kargs):
    # Data specific preprocessing of cell info
    file_name = 'data/GSE110894_CellInfo.xlsx' # change it to the name of your excel file
    df_cellinfo = read_excel(file_name,header=3)
    df_cellinfo=df_cellinfo.dropna(how="all")
    df_cellinfo = df_cellinfo.fillna(method='pad')
    well_post = ["_"+wp.split("=")[0] for wp in df_cellinfo.loc[:,'Well position']]
    inversindex = df_cellinfo.loc[:,'Plate#']+well_post
    inversindex.name = 'Index'
    df_cellinfo.index = inversindex
    obs_merge = pd.merge(adata.obs,df_cellinfo,left_index=True,right_index=True,how='left')
    adata.obs = obs_merge
    sensitive = [int(row.find("RESISTANT")==-1) for row in obs_merge.loc[:,"Sample name"]]
    adata.obs['sensitive'] = sensitive

    sens_ = ['Resistant' if (row.find("RESISTANT")!=-1) else 'Sensitive' for row in obs_merge.loc[:,"Sample name"]]
    adata.obs['sensitivity'] = sens_


    pval = 0.05
    n_genes = 50
    if "pval_thres" in kargs:
        pval=kargs['pval_thres']
    if "num_de" in kargs:
        n_genes = kargs['num_de']
    adata = de_score(adata=adata,clustername="sensitivity",pval=pval,n=n_genes)    
    return adata


def process_112274(adata,**kargs):
    obs_names = adata.obs.index
    annotation_dict = {}
    for section in [0,1,2,3]:
        svals = [index.split("_")[section] for index in obs_names]
        annotation_dict["name_section_"+str(section+1)] = svals
    df_annotation=pd.DataFrame(annotation_dict,index=obs_names)
    adata.obs=df_annotation

    sensitive = [int(row.find("parental")!=-1) for row in df_annotation.loc[:,"name_section_2"]]
    adata.obs['sensitive'] = sensitive

    sens_ = ['Resistant' if (row.find("parental")==-1) else 'Sensitive' for row in df_annotation.loc[:,"name_section_2"]]
    adata.obs['sensitivity'] = sens_


    pval = 0.05
    n_genes = 50
    if "pval_thres" in kargs:
        pval=kargs['pval_thres']
    if "num_de" in kargs:
        n_genes = kargs['num_de']
    adata = de_score(adata=adata,clustername="sensitivity",pval=pval,n=n_genes)    

    return adata

def process_116237(adata,**kargs):
    obs_names = adata.obs.index
    annotation_dict = {}
    for section in [0,1,2]:
        svals = [re.split('_|\.',index)[section] for index in obs_names]
        annotation_dict["name_section_"+str(section+1)] = svals  

    return adata

def process_140440(adata,**kargs):
    # Data specific preprocessing of cell info
    file_name = 'data/GSE140440/Annotation.txt' # change it to the name of your excel file
    df_cellinfo = pd.read_csv(file_name,header=None,index_col=0,sep="\t")
    sensitive = [int(row.find("Res")==-1) for row in df_cellinfo.iloc[:,0]]
    adata.obs['sensitive'] = sensitive

    sens_ = ['Resistant' if (row.find("Res")!=-1) else 'Sensitive' for row in df_cellinfo.iloc[:,0]]
    adata.obs['sensitivity'] = sens_


    pval = 0.05
    n_genes = 50
    if "pval_thres" in kargs:
        pval=kargs['pval_thres']
    if "num_de" in kargs:
        n_genes = kargs['num_de']
    adata = de_score(adata=adata,clustername="sensitivity",pval=pval,n=n_genes)    
    return adata

def process_129730(adata,**kargs):
    #Data specific preprocessing of cell info
    # sensitive = [ 1 if row in [''] \
    #                 for row in adata.obs['sample']]
    sensitive = [ 1 if (row <=9) else 0 for row in adata.obs['sample'].astype(int)]
    adata.obs['sensitive'] = sensitive
    sens_ = ['Resistant' if (row >9) else 'Sensitive' for row in adata.obs['sample'].astype(int)]
    adata.obs['sensitivity'] = sens_


    pval = 0.05
    n_genes = 50
    if "pval_thres" in kargs:
        pval=kargs['pval_thres']
    if "num_de" in kargs:
        n_genes = kargs['num_de']
    adata = de_score(adata=adata,clustername="sensitivity",pval=pval,n=n_genes)    
    return adata
    
def process_149383(adata,**kargs):
    # Data specific preprocessing of cell info
    file_name = 'data/GSE149383/erl_total_2K_meta.csv' # change it to the name of your excel file
    df_cellinfo = pd.read_csv(file_name,header=None,index_col=0)
    sensitive = [int(row.find("res")==-1) for row in df_cellinfo.iloc[:,0]]
    adata.obs['sensitive'] = sensitive

    sens_ = ['Resistant' if (row.find("res")!=-1) else 'Sensitive' for row in df_cellinfo.iloc[:,0]]
    adata.obs['sensitivity'] = sens_


    pval = 0.05
    n_genes = 50
    if "pval_thres" in kargs:
        pval=kargs['pval_thres']
    if "num_de" in kargs:
        n_genes = kargs['num_de']
    adata = de_score(adata=adata,clustername="sensitivity",pval=pval,n=n_genes)    
    return adata

def integrated_gradient_check(net,input,target,adata,n_genes,target_class=1,test_value="expression",save_name="feature_gradients",batch_size=100):
        ig = IntegratedGradients(net)
        attr, delta = ig.attribute(input,target=target_class, return_convergence_delta=True,internal_batch_size=batch_size)
        attr = attr.detach().cpu().numpy()
        adata.var['integrated_gradient_sens_class'+str(target_class)] = attr.mean(axis=0)

        sen_index = (target == 1)
        res_index = (target == 0)

        # Add col names to the DF
        attr = pd.DataFrame(attr, columns = adata.var.index)

        # Construct attr as a dafaframe
        df_top_genes = adata.var.nlargest(n_genes,"integrated_gradient_sens_class"+str(target_class),keep='all')
        df_tail_genes = adata.var.nsmallest(n_genes,"integrated_gradient_sens_class"+str(target_class),keep='all')
        list_topg = df_top_genes.index 
        list_tailg = df_tail_genes.index 

        top_pvals = []
        tail_pvals = []

        if(test_value=='gradient'):
            feature_sens = attr[sen_index]
            feature_rest = attr[res_index]
        else:        
            expression_norm = input.detach().cpu().numpy()
            expression_norm = pd.DataFrame(expression_norm, columns = adata.var.index)
            feature_sens = expression_norm[sen_index]
            feature_rest = expression_norm[res_index]

        for g in list_topg:
            f_sens = feature_sens.loc[:,g]
            f_rest = feature_rest.loc[:,g]
            stat,p =  mannwhitneyu(f_sens,f_rest)
            top_pvals.append(p)

        for g in list_tailg:
            f_sens = feature_sens.loc[:,g]
            f_rest = feature_rest.loc[:,g]
            stat,p =  mannwhitneyu(f_sens,f_rest)
            tail_pvals.append(p)

        df_top_genes['pval']=top_pvals
        df_tail_genes['pval']=tail_pvals

        df_top_genes.to_csv("saved/results/top_genes_class" +str(target_class)+ save_name + '.csv')
        df_tail_genes.to_csv("saved/results/top_genes_class" +str(target_class)+ save_name + '.csv')

        return adata,attr,df_top_genes,df_tail_genes

def integrated_gradient_differential(net,input,target,adata,n_genes=None,target_class=1,clip="abs",save_name="feature_gradients",ig_pval=0.05,ig_fc=1,method="wilcoxon",batch_size=100):
        
        # Caculate integrated gradient
        ig = IntegratedGradients(net)

        df_results = {}

        attr, delta = ig.attribute(input,target=target_class, return_convergence_delta=True,internal_batch_size=batch_size)
        attr = attr.detach().cpu().numpy()

        if clip == 'positive':
            attr = np.clip(attr,a_min=0,a_max=None)
        elif clip == 'negative':
            attr = abs(np.clip(attr,a_min=None,a_max=0))
        else:
            attr = abs(attr)

        igadata= sc.AnnData(attr)
        igadata.var.index = adata.var.index
        igadata.obs.index = adata.obs.index

        igadata.obs['sensitive'] = target
        igadata.obs['sensitive'] = igadata.obs['sensitive'].astype('category')

        sc.tl.rank_genes_groups(igadata, 'sensitive', method=method,n_genes=n_genes)

        for label in [0,1]:

            try:
                df_degs = ut.get_de_dataframe(igadata,label)
                df_degs = df_degs.loc[(df_degs.pvals_adj<ig_pval) & (df_degs.logfoldchanges>=ig_fc)]
                df_degs.to_csv("saved/results/DIG_class_" +str(target_class)+"_"+str(label)+ save_name + '.csv')

                df_results[label]= df_degs
            except:
                logging.warning("Only one class, no two calsses critical genes")

        return adata,igadata,list(df_results[0].names),list(df_results[1].names)

def de_score(adata,clustername,pval=0.05,n=50,method="wilcoxon",score_prefix=None):
    try:
        sc.tl.rank_genes_groups(adata, clustername, method=method,use_raw=True)
    except:
        sc.tl.rank_genes_groups(adata, clustername, method=method,use_raw=False)
    # Cluster de score
    for cluster in set(adata.obs[clustername]):
        df = ut.get_de_dataframe(adata,cluster)
        select_df = df.iloc[:n,:]
        if pval!=None:
            select_df = select_df.loc[df.pvals_adj < pval]
        sc.tl.score_genes(adata, select_df.names,score_name=str(cluster)+"_score" )
    return adata

def plot_loss(report,path="figures/loss.pdf",set_ylim=False):

    train_loss = []
    val_loss = []


    epochs = int(len(report)/2)
    print(epochs)

    score_dict = {'train':train_loss,'val':val_loss}

    for phrase in ['train','val']:
        for i in range(0,epochs):
            score_dict[phrase].append(report[(i,phrase)])
    plt.close()
    plt.clf()
    x = np.linspace(0, epochs, epochs)
    plt.plot(x,val_loss, '-g', label='validation loss')
    plt.plot(x,train_loss,':b', label='trainiing loss')
    plt.legend(["validation loss", "trainiing loss"], loc='upper left')
    if set_ylim!=False:
        plt.ylim(set_ylim)
    plt.savefig(path)
    plt.close()

    return score_dict

def integrated_gradient_differential(net,input,target,adata,n_genes=None,target_class=1,clip="abs",save_name="feature_gradients",ig_pval=0.05,ig_fc=1,method="wilcoxon",batch_size=100):
        
        # Caculate integrated gradient
        ig = IntegratedGradients(net)

        df_results = {}

        attr, delta = ig.attribute(input,target=target_class, return_convergence_delta=True,internal_batch_size=batch_size)
        attr = attr.detach().cpu().numpy()

        if clip == 'positive':
            attr = np.clip(attr,a_min=0,a_max=None)
        elif clip == 'negative':
            attr = abs(np.clip(attr,a_min=None,a_max=0))
        else:
            attr = abs(attr)

        igadata= sc.AnnData(attr)
        igadata.var.index = adata.var.index
        igadata.obs.index = adata.obs.index

        igadata.obs['sensitive'] = target
        igadata.obs['sensitive'] = igadata.obs['sensitive'].astype('category')

        sc.tl.rank_genes_groups(igadata, 'sensitive', method=method,n_genes=n_genes)

        for label in [0,1]:

            try:
                df_degs = ut.get_de_dataframe(igadata,label)
                df_degs = df_degs.loc[(df_degs.pvals_adj<ig_pval) & (df_degs.logfoldchanges>=ig_fc)]
                df_degs.to_csv("saved/results/DIG_class_" +str(target_class)+"_"+str(label)+ save_name + '.csv')

                df_results[label]= df_degs
            except:
                logging.warning("Only one class, no two calsses critical genes")

        return adata,igadata,list(df_results[0].names),list(df_results[1].names)

