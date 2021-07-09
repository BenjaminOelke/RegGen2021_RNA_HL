import os.path
import pyranges as pr
import pandas as pd
import time
import plotnine as p9
import numpy as np
import pickle
import kipoiseq
#import tensorflow
from palettable.matplotlib import matplotlib
from plotnine import geom_boxplot,geom_point
from sklearn.linear_model import Lasso,MultiTaskLassoCV
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,r2_score,explained_variance_score,mean_squared_error
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

iris = sns.load_dataset("iris")
species = iris.pop("species")
lut = dict(zip(species.unique(), "rbg"))
row_colors = species.map(lut)





start = time.time()

#load data and prepare dataframes and lists
df_hl = pd.read_csv('genomic_sequence_plus_features_hl_all_tissues.csv',index_col=0)
df_hl = df_hl.drop(columns=["TGA","TAA","TAG"]) #drop stop codons  #drop everything else ,"log_3_utr_length","log_5_utr_length","log_cds_length","gc_content_3_utr","gc_content_5_utr","gc_content_cds"
df_hl = df_hl.drop(columns=['cds', '3_utr', '5_utr', 'all'])

tissue_hl_e = pd.read_csv('genomic_sequence_plus_features_hl_all_tissues.csv', index_col=False)
tissue_hl_e.rename(columns={'Unnamed: 0':'Name'},inplace=True)

codons = ['AAA', 'AAC','AAG', 'AAT', 'ACA', 'ACC', 'ACG', 'ACT', 'AGA', 'AGC', 'AGG', 'AGT','ATA', 'ATC', 'ATG', 'ATT', 'CAA', 'CAC', 'CAG', 'CAT', 'CCA', 'CCC','CCG', 'CCT', 'CGA', 'CGC', 'CGG', 'CGT', 'CTA', 'CTC', 'CTG', 'CTT','GAA', 'GAC', 'GAG', 'GAT', 'GCA', 'GCC', 'GCG', 'GCT', 'GGA', 'GGC',
            'GGG', 'GGT', 'GTA', 'GTC', 'GTG', 'GTT', 'TAC', 'TAT','TCA', 'TCC', 'TCG', 'TCT', 'TGC', 'TGG', 'TGT', 'TTA', 'TTC','TTG', 'TTT']

gc = ["gc_content_3_utr",  "gc_content_5_utr",  "gc_content_cds"]

df_rbp = pd.read_csv("./results/hg19_k562_all_gene_body_RBP.txt", index_col=0)
df_rbp = df_rbp.drop(columns="gene_length")

df_mirna = pd.read_csv("./results/Number_RNA_family.csv",index_col=0)
df_mirna["mirna_sum"] = df_mirna.sum(axis=1)
df_mirna = df_mirna.set_index("id")

utr_5_txt=pd.read_csv('results/hgTables_5utr', delimiter='\t')
utr_5_txt = utr_5_txt.loc[utr_5_txt.loc[:, ['#name','energy']].dropna().index]
utr_5_txt.rename(columns={'energy':'energy_5utr'},inplace=True)
utr_5_txt.rename(columns={'#name':'Name'},inplace=True)
utr_5_txt.rename(columns={'#name':'Name'},inplace=True)
utr_5_txt=utr_5_txt.drop(['seq','fold'], axis=1)

tissue_hl_E= tissue_hl_e
tissue_hl_E= pd.concat([tissue_hl_E,utr_5_txt['energy_5utr']],axis=1,join='inner').set_index("Name")
df_enrg= pd.DataFrame(tissue_hl_E["energy_5utr"])

#log codon frequencies (did not imporve performance but made it far worse)
#df_hl[codons]=np.log(df_hl[codons].replace(to_replace=0.0,value=0.00000000001))

#get a list of all tissues
tissues = df_hl.columns[:49].tolist()

#set the test chromosomes
chrom_test=['chr1','chr8','chr9','chr2', 'chr3','chr4']

#transform half life from absolute values to relative to the mean
means = []
for row in df_hl.itertuples():
    mean = np.nanmean(row[1:50])
    means.append(mean)

df_hl.insert(loc=49, column='Mean', value=means)

# disable this loop to predict absolute values, activate it to predict relative to the mean #################### SWITCH ABS/REL
for tissue in tissues:
    df_hl[tissue] = df_hl[tissue]-df_hl['Mean']
#add column to predict mean hl
tissues.append("Mean")
index = df_hl.index

rbp,energy,mirna = [False,False,True]

if rbp:
    #add rbp
    transcripts = pickle.load(open("./pickles/t2g.p", "rb"))
    transcripts['gene_id'] = transcripts['gene_id'].str.split('.').str[0]
    transcripts.set_index('transcript_id', inplace=True)
    df_hl['gene_id'] = df_hl.index.map(transcripts["gene_id"])
    df_hl = df_hl.merge(df_rbp, on="gene_id", how="left")
    df_hl = df_hl.drop(columns=("gene_id")).set_index(index)

#add 5"UTR energy
if energy:
    df_hl = pd.concat([df_hl,df_enrg],axis=1)

#add mirna
if mirna:
    df_hl = pd.concat([df_hl,df_mirna],axis=1)
# train test split by chromosome
train = df_hl[~df_hl['chromosome'].isin(chrom_test)]
train = train.drop(columns=['chromosome'])

test = df_hl[df_hl['chromosome'].isin(chrom_test)]
test = test.drop(columns=['chromosome'])

X_train = train.drop(columns=tissues, axis=1).fillna(0)
X_test = test.drop(columns=tissues, axis=1).fillna(0)
y_train = train[tissues].fillna(0)
y_test = test[tissues].fillna(0)

MTL = MultiTaskLassoCV()
MTL.fit(X_train,y_train)
y_pred = MTL.predict(X_test)
# regression
y_pred = pd.DataFrame(y_pred,index=y_test.index,columns=tissues)

df_coef = pd.DataFrame(MTL.coef_,index=tissues,columns=X_test.columns).T

df_test = df_coef

xpv_mt ={}

for tissue in tissues:
    xpv_t = explained_variance_score(y_test[tissue], y_pred[tissue])
    xpv_mt[tissue] = xpv_t
    df_coef[tissue] = df_coef[tissue]-df_coef[tissue].mean()

xpv_mt = pd.DataFrame.from_dict(xpv_mt,orient='index')
df_csv = pd.concat([df_coef.T,xpv_mt],axis=1)
df_csv = df_csv.rename(columns={0:"exp_var_sc"})
df_csv.to_csv("results/relative/csv/mtl_base_mirna.csv")
xpv_mt = xpv_mt.reset_index()
xpv_mt = xpv_mt.rename(columns={"index":"tissue",0:"exp_var_sc"})
coef_plot = sns.clustermap(df_coef,center = 0,cbar_pos=(0.92, 0.04, .015, .2),yticklabels=True,xticklabels=True,robust = True,cmap = "RdBu_r")
coef_plot.ax_heatmap.set_yticklabels(coef_plot.ax_heatmap.get_ymajorticklabels(), fontsize=3)

for tick_label in coef_plot.ax_heatmap.axes.get_yticklabels():
    tick_text = tick_label.get_text()
    if tick_text == 'energy_5utr':
        tick_label.set_color("red")
    if tick_text in df_rbp.columns:
        tick_label.set_color("green")
    if tick_text in df_mirna.columns:
        tick_label.set_color("blue")
    if tick_text in gc:
        tick_label.set_color("orange")


plt.savefig("results/relative/multi_task/heat_coef_mtl_base_mirna.png")
plt.clf()
xpv_plot = p9.ggplot(xpv_mt, p9.aes('tissue', 'exp_var_sc')) + p9.geom_col() + p9.theme(
axis_text_x=p9.element_text(angle=90,size=5)) + p9.ylim(-0.02,0.2)
xpv_plot.save(filename="results/relative/multi_task/xpv_mtl_base_mirna.jpg")
plt.clf()
end = time.time()
print(end - start)

