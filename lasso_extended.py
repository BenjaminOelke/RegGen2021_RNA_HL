import os.path

import pandas as pd
import time
import plotnine as p9
import numpy as np
import pickle
import kipoiseq
#import tensorflow
from palettable.matplotlib import matplotlib
from plotnine import geom_boxplot,geom_point
from sklearn.linear_model import Lasso,MultiTaskLasso
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,r2_score,explained_variance_score,mean_squared_error
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def lasso_regression(incl_rbp,incl_mirna,rdc_transcripts,drop_codons):

    tissue_results = []

    for i in range(len(tissues)):
        print("[" + str(i) + "]")

        start = time.time()
        df_tiss = df_hl_enc.loc[df_hl_enc['tissue'] == i]
        df_tiss.index = df_hl.index
        tissue_results.append(per_tissue(df_tiss,df_rbp,df_mirna ,i,incl_rbp,incl_mirna,rdc_transcripts,drop_codons))
        end = time.time()
        print(end - start)

    print()
    xvar = list(list(zip(*tissue_results))[0])
    mse = list(list(zip(*tissue_results))[1])


    var = {'exp_var_sc': xvar, 'tissue': tissues}
    m2e = {'%root_mean_sq_err': mse, 'tissue': tissues}
    pl_var = pd.DataFrame(var)
    pl_m2e = pd.DataFrame(m2e)

    results = [tissue_results,pl_var,pl_m2e]



    #pickle dump
    f_path = "./pickles/"+file_name(incl_rbp,incl_mirna,rdc_transcripts,drop_codons,False)+".p"
    pickle.dump(results, open(f_path, "wb"))


def per_tissue(df_hl,df_rbp,df_mirna,i,rbps,mirnas,rdcs,dr_cdn):

    df1 = df_hl.drop(columns=['cds', '3_utr', '5_utr', 'all','tissue'])
    df1 = df1.drop(columns=codons)


    df1 = df1.loc[df1.loc[:, ['half_life']].dropna().index]
    df1 = df1.fillna(0)

    #gradient descent to tune hyperparameters
    X = df1.drop(columns=['half_life','chromosome'], axis=1)
    y = df1['half_life']
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    model = Lasso(max_iter=5000)
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    param = {
        'alpha':[.00001, 0.0001,0.001, 0.01],
        'fit_intercept':[True,False],
        'normalize':[True,False],
        'positive':[True,False],
        'selection':['cyclic','random'],
        }

    search = GridSearchCV(model, param, scoring='neg_mean_absolute_error', n_jobs=-1, cv=cv)
    result = search.fit(X, y)

    df2 = df_hl.drop(columns=['cds', '3_utr', '5_utr','all'])
    if dr_cdn:
        df2 = df2.drop(columns=codons)


    if rbp:
        df_rbp = df_rbp.drop(columns=("chromosome"))

        if rdcs:
            a_series = (df_rbp != 0).any(axis=1)
            df_rbp = df_rbp.loc[a_series]

        df2 = df2.loc[df_rbp.index]
        df2 = pd.concat([df2, df_rbp], axis=1)


    if mirna:
        if rdcs:
            a_series = (df_mirna != 0).any(axis=1)
            df_mirna = df_rbp.loc[a_series]
        df2 = df2.loc[df_mirna.index]
        df2 = pd.concat([df2,df_mirna],axis=1)

    df2 = df2.loc[df1.loc[:, ['half_life']].dropna().index]
    df2 = df2.fillna(0)


    #train test split by chromosome
    train = df2[~df2['chromosome'].isin(chrom_test)]
    train = train.drop(columns=['chromosome','tissue'])

    test = df2[df2['chromosome'].isin(chrom_test)]
    test = test.drop(columns=['chromosome','tissue'])

    X_train = train.drop("half_life", axis=1)
    X_test = test.drop("half_life", axis=1)
    y_train = train["half_life"]
    y_test = test["half_life"]

    #regression
    model = Lasso(alpha=result.best_params_['alpha'], fit_intercept=result.best_params_['fit_intercept'], normalize=result.best_params_['normalize'], positive=result.best_params_['positive'], selection=result.best_params_['selection'])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)


    msqe = mean_squared_error(y_test,y_pred,squared=False)/((max(y_test)-min(y_test))/100)
    xpv = explained_variance_score(y_test, y_pred)
    print(xpv)

    #extracting the model coefficients and map them to tissue
    coef = pd.DataFrame(model.coef_).T
    coef.columns = X_test.columns
    coef.index = [tissues[i]]

    #extract predicted and corresponding true values and pack them in a df

    y_test = y_test.to_numpy()
    vs = np.ma.row_stack([y_pred, y_test])
    vs = pd.DataFrame(vs)
    vs.index = ["pred","truth"]

    return [xpv,msqe,vs,coef]

def reduce_feature_to_transcripts_hit_on_chr(df):

    index_chr_dict = {}

    for i in df.index:
        index_chr_dict[i] = df.loc[i]["chromosome"]

    df = df.drop(columns="chromosome")
    a_series = (df != 0).any(axis=1)
    df = df.loc[a_series]
    df = df.assign(chromosome="zero")


    for i in df.index:
        df["chromosome"][i]  = index_chr_dict[i]

    return(df)

def get_balanced_feature_df(df_feat,df_tiss,):

    #rmv hit transcripts
    df_tiss = df_tiss.drop(df_feat.index)
    print()

def file_name(incl_rbp,incl_mirna,rdc_transcripts,drop_codons,mk_path):

    feature = "_base"
    if incl_rbp:
        feature = "_rbp"
    if incl_mirna:
        feature = "_mirna"

    transcripts = "_t_all"
    if rdc_transcripts:
        transcripts = "_t_rdc"
    cdns = "_cdn"
    if drop_codons:
        cdns = "_cdn_d"

    file_name = "lasso" + feature + transcripts + cdns

    if mk_path:
        path = "results/relative/"+feature
        if not os.path.isdir(path):
            os.mkdir(path)
            #os.mkdir(path+"/pred_vs_truth")
        return path

    return file_name

def plot_results(res,incl_rbp,incl_mirna,rdc_transcripts,drop_codons):

    path = file_name(incl_rbp,incl_mirna,rdc_transcripts,drop_codons,True)
    name = file_name(incl_rbp,incl_mirna,rdc_transcripts,drop_codons,False)
    lasso_list = res[0]
    df_xpv = res[1]
    df_mse = res[2]

    # lasso_list[i] has xpv at lasso_list[i][0] ,mse at [1], vs at [2] and  coef at [3]

    xpv_plot = p9.ggplot(df_xpv, p9.aes('tissue', 'exp_var_sc')) + p9.geom_col() + p9.theme(
        axis_text_x=p9.element_text(angle=90))
    xpv_plot.save(filename=path+"/xpv_"+name+".jpg")

    mse_plot = p9.ggplot(df_mse, p9.aes('tissue', '%root_mean_sq_err')) + p9.geom_col() + p9.theme(
        axis_text_x=p9.element_text(angle=90))
    mse_plot.save(filename=path+"/mse_"+name+".jpg")

    df_coef = pd.DataFrame(columns=lasso_list[0][3].columns)

    for i in range(len(tissues)):
        #cur_lasso = lasso_list[i][2]
        #sns.scatterplot(data=cur_lasso.T, x="pred", y="truth").set(title=tissues[i])

        #plt.savefig("results/relative/pred_vs_truth/" + tissues[i])  ############## CHANGE HERE
        #plt.clf()

        # built the coefficient dataframe
        df_coef = df_coef.append(lasso_list[i][3])

    df_coef = df_coef.replace(to_replace=0.0, value=0.00000000001)
    coef_plot = sns.heatmap(df_coef)

    plt.savefig(path+"/heat_coef_"+name+".png")
    plt.clf()

    rsc = df_coef.drop(columns=["TGA", "TAA", "TAG"])
    rsc_plot = sns.heatmap(rsc)
    plt.savefig(path+"/heat_coef_"+name+".png")



#MAIN####################################MAIN#########################################################MAIN##############################################################

#load data and prepare dataframes and lists
df_hl = pd.read_csv('genomic_sequence_plus_features_hl_all_tissues.csv',index_col=0)
df_hl = df_hl.drop(columns=["TGA","TAA","TAG"]) #drop stop codons

codons = ['AAA', 'AAC','AAG', 'AAT', 'ACA', 'ACC', 'ACG', 'ACT', 'AGA', 'AGC', 'AGG', 'AGT','ATA', 'ATC', 'ATG', 'ATT', 'CAA', 'CAC', 'CAG', 'CAT', 'CCA', 'CCC','CCG', 'CCT', 'CGA', 'CGC', 'CGG', 'CGT', 'CTA', 'CTC', 'CTG', 'CTT','GAA', 'GAC', 'GAG', 'GAT', 'GCA', 'GCC', 'GCG', 'GCT', 'GGA', 'GGC',
            'GGG', 'GGT', 'GTA', 'GTC', 'GTG', 'GTT', 'TAC', 'TAT','TCA', 'TCC', 'TCG', 'TCT', 'TGC', 'TGG', 'TGT', 'TTA', 'TTC','TTG', 'TTT']

df_mirna = pd.read_csv("./results/Number_RNA_family.csv",index_col=0)
df_mirna["sum"] = df_mirna.sum(axis=1)
df_mirna = df_mirna.set_index("id")

df_rbp = pd.read_csv("./results/rbp_utr_gencode_19_binding.csv",index_col=0)
df_rbp["sum"] = df_rbp.sum(axis=1)


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

df_hl_enc =df_hl


# encode tissues in a single coloumn
for i in range (len(tissues)):  #49 tissue columns + 1 mean at the end

  df_hl_enc=df_hl_enc.rename(columns={df_hl.columns[i]:i})


#make a row for each hl in tissue/gene combination
df_hl_enc=pd.melt(df_hl_enc, id_vars=df_hl_enc.columns[len(tissues):], value_vars=df_hl_enc.columns[0:len(tissues)], var_name="tissue", value_name="half_life")

#regression needs to be done only once for a given feature / prediction(abs/rel) as its results are stored in a pkl
#lines where path changes might be neccessary are marked with "CHANGE HERE"


rbp = False
mirna = False
rdc = False
cdns = True

lasso_regression(rbp,mirna,rdc,cdns)
f_path = "./pickles/"+file_name(rbp,mirna,rdc,cdns,False)+".p"
results = pickle.load(open(f_path, "rb"))
plot_results(results,rbp,mirna,rdc,cdns)



'''
 
 ################################################################################
# get counts of transcripts per chromosome
chr_hl = {}
github

for t in set(df_hl["chromosome"]):
    n = df_hl["chromosome"].value_counts()[t]
    chr_hl[t] = n

# get counts of rbp bs found per chromosome
chr_rbp = {}
df_temp = df_rbp.loc[df_rbp["sum"]>0]

for t in set(df_temp["chromosome"]):
    n = df_temp["chromosome"].value_counts()[t]
    chr_rbp[t] = n
print()
# get counts of mirna bs found per chromosome
chr_mirna = {}

df_temp = df_mirna.loc[df_mirna["sum"]>0]
df_temp = df_hl.loc[set(df_temp["id"]),:]
print()
for t in set(df_temp["chromosome"]):
    n = df_temp["chromosome"].value_counts()[t]
    chr_mirna[t] = n

print()
df_chr = pd.DataFrame.from_dict([chr_hl,chr_rbp,chr_mirna])
df_chr["Factor"] = []
    
df_chr.T.plot.bar()
####################################################################################################
'''