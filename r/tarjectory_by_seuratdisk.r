library(Seurat)
library(SeuratDisk)
library(SeuratWrappers)
library(monocle3)
library(ggplot2)
library(dplyr) 
source("plotc.r")

data <- "GSE117872_HN120"
#subset <- FALSE

#subset <- c("HN120P","HN120PCR")
subset <- c("HN120M","HN120MCR")
#subset <- c("HN137M","HN137MCR")
#subset <- c("HN137P","HN120CR")


ct.key <- function(data) {
  if(data=="GSE110894"){
    return("Sample.name")
  }else if(data=="GSE117872_HN120"){
    return("cell_color")
  }else if(data=="GSE117872_HN137"){
    return("cell_color")
  }else{
    return(leiden)
  }
}

# Path of H5AD data
h5adname <- Sys.glob(paste("review/*",data,"*h5ad", sep = ""))[1]

Convert(h5adname, dest = "h5seurat",verbose = TRUE, overwrite = TRUE)

h5seuratname <- Sys.glob(paste("review/*",data,"*h5seurat", sep = ""))[1]

h5ad <- LoadH5Seurat(h5seuratname)

`%ni%` <- Negate(`%in%`)



if(data=="GSE110894"){
  h5ad <- subset(h5ad, subset=Sample.name %ni% c("EMPTY","EMPTY ","101 CELL CONTROL"))
  levels(h5ad@meta.data$Sample.name) <- c("NONE1","NONE2","NONE3","RESISTANT","WITHDRAW","SENSITIVE","SENSITIVE")
  
}else if(subset!=FALSE){
  h5ad <- subset(h5ad, subset=cell_color %in% subset)
}

cds <- as.cell_data_set(h5ad)
cds <- estimate_size_factors(cds)
cds@rowRanges@elementMetadata@listData[["gene_short_name"]] <- rownames(h5ad)

# colnames(cds@colData)[1] <- "cell_type"
colnames(cds@colData)[colnames(cds@colData)=="sens_preds"] <- "response_probability"

#cds <- preprocess_cds(cds, num_dim = 50,method = 'PCA')
cds <- reduce_dimension(cds)


plot_cells(cds, label_groups_by_cluster=FALSE, cell_size = 0.6, color_cells_by = ct.key(data))


cds <- cluster_cells(cds)
#plot_cells(cds, color_cells_by = "partition")


cds <- learn_graph(cds)


plot_cells(cds,
           color_cells_by = "cell_color",
           label_groups_by_cluster=FALSE,
           label_leaves=FALSE,
           cell_size = 0.6,
           label_branch_points=FALSE)

ggsave(filename  = paste("results/",data,subset[1],"_cell_color.svg", sep = "") , width = 3, height = 3, device='svg', dpi=400)


# Path of the critical genes name list
f_sg <- Sys.glob(paste("data/*",data,"*s*csv", sep = ""))[1]
f_rg <- Sys.glob(paste("data/*",data,"*r*csv", sep = ""))[1]


df_sg <- read.csv(file = f_sg,header = FALSE)
df_rg <- read.csv(file = f_rg,header = FALSE)

SC_genes <- c(head(df_sg,5))[[1]]
RC_genes <- c(head(df_rg,5))[[1]]



plotc(cds,genes = SC_genes,   
      label_groups_by_cluster=FALSE,
      label_leaves=FALSE,
      label_roots=FALSE,
      label_branch_points=FALSE,
      cell_size = 0.6,
      graph_label_size=1.5) +  theme(legend.position="top")


ggsave(filename  = paste("results/",data,subset[1],"_s_cg.svg", sep = "") , width = 9, height = 7, device='svg', dpi=400)


plotc(cds,genes = RC_genes,   
      label_groups_by_cluster=FALSE,
      label_leaves=FALSE,
      label_roots=FALSE,
      label_branch_points=FALSE,
      cell_size = 0.6,
      graph_label_size=1.5) +  theme(legend.position="top")

ggsave(filename  = paste("results/",data,subset[1],"_r_cg.svg", sep = "") , width = 9, height = 7, device='svg', dpi=400)

cds <- order_cells(cds)

# Open a pdf file
# pdf("rplot.svg", width=5, height=5) 
plot_cells(cds,
           color_cells_by = "response_probability",
           label_cell_groups=FALSE,
           label_leaves=FALSE,
           label_branch_points=FALSE,
           cell_size = 0.6,
           graph_label_size=1.5) +  theme(legend.position="top")
# Open a pdf file
# dev.off() 
ggsave(filename  = paste("results/",data,subset[1],"_response_prob.svg", sep = ""), width = 3, height = 3, device='svg', dpi=400)

cds_sub <- choose_graph_segments(cds)

plot_cells(cds,
           color_cells_by = "pseudotime",
           label_cell_groups=FALSE,
           label_leaves=FALSE,
           label_branch_points=FALSE,
           graph_label_size=1.5) +  theme(legend.position="top")

ggsave(filename  = paste("results/",data,subset[1],"_pseudotime.svg", sep = ""), width = 3, height = 3, device='svg', dpi=400)


DS_lineage_cds <- cds[rowData(cds)$gene_short_name %in% SC_genes,]

plot_genes_in_pseudotime(DS_lineage_cds,
                         color_cells_by=ct.key(data),
                         min_expr=0.5)

ggsave(filename  = paste("results/",data,subset[1],"_s_linage.svg", sep = ""), width = 3, height = 3, device='svg', dpi=400)


DR_lineage_cds <- cds[rowData(cds)$gene_short_name %in% RC_genes,]

plot_genes_in_pseudotime(DR_lineage_cds,
                         color_cells_by=ct.key(data),
                         min_expr=0.5)

ggsave(filename  = paste("results/",data,subset[1],"_r_linage.svg", sep = ""), width = 3, height = 3, device='svg', dpi=400)

plot_genes_in_pseudotime(DR_lineage_cds,
                         color_cells_by="response_probability",
                         min_expr=0.5)

ggsave(filename  = paste("results/",data,subset[1],"_rp_linage.svg", sep = ""), width = 3, height = 3, device='svg', dpi=400)



# rowData(cds)$gene_name <- rownames(cds)
# rowData(cds)$gene_short_name <- rowData(cds)$gene_name