import pandas as pd
import time
import plotnine as p9
import numpy as np
import pickle
#import kipoiseq
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

def lasso_regression():


    tissue_results = []

    for i in range(len(tissues)):
        print("[" + str(i) + "]")

        start = time.time()
        df_tiss = df_hl_enc.loc[df_hl_enc['tissue'] == i]

        tissue_results.append(per_tissue(df_tiss,df_rbp,df_mirna ,i,True,False))
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
    pickle.dump(results, open("./pickles/lasso_relative_per_tissue_base.p", "wb"))                ############## CHANGE HERE








def per_tissue(df_hl,df_rbp,df_mirna,i,rbp,mirna):

    df1 = df_hl.drop(columns=['cds', '3_utr', '5_utr', 'all','tissue'])
    df1 = df1.drop(columns=codons)

    if rbp:
        df_rbp = df_rbp.drop(columns=("chromosome"))
        df1 = pd.concat([df1,df_rbp],axis=1)

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

    df2 = df.drop(columns=['cds', '3_utr', '5_utr','all'])
    df2 = df2.loc[df1.loc[:, ['half_life']].dropna().index]
    df2 = df2.fillna(0)
    X = df2.drop('half_life', axis=1)
    y = df2['half_life']

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

#MAIN####################################MAIN#########################################################MAIN

#load data and prepare dataframes and lists
df_hl = tissue_hl = pd.read_csv('genomic_sequence_plus_features_hl_all_tissues.csv', index_col=0)


df_mirna = pd.read_csv("./results/Number_RNA_family.csv",index_col=0)
df_mirna["sum"] = df_mirna.sum(axis=1)

df_rbp = pd.read_csv("./results/rbp_utr_gencode_19_binding.csv",index_col=0)
df_rbp["sum"] = df_rbp.sum(axis=1)

codons = ['AAA', 'AAC','AAG', 'AAT', 'ACA', 'ACC', 'ACG', 'ACT', 'AGA', 'AGC', 'AGG', 'AGT','ATA', 'ATC', 'ATG', 'ATT', 'CAA', 'CAC', 'CAG', 'CAT', 'CCA', 'CCC','CCG', 'CCT', 'CGA', 'CGC', 'CGG', 'CGT', 'CTA', 'CTC', 'CTG', 'CTT','GAA', 'GAC', 'GAG', 'GAT', 'GCA', 'GCC', 'GCG', 'GCT', 'GGA', 'GGC',
            'GGG', 'GGT', 'GTA', 'GTC', 'GTG', 'GTT', 'TAA', 'TAC', 'TAG', 'TAT','TCA', 'TCC', 'TCG', 'TCT', 'TGA', 'TGC', 'TGG', 'TGT', 'TTA', 'TTC','TTG', 'TTT']

#log codon frequencies (did not imporve performance but made it far worse)
#df_hl[codons]=np.log(df_hl[codons].replace(to_replace=0.0,value=0.00000000001))

#get a list of all tissues
tissues = tissue_hl.columns[:49].tolist()

#set the test chromosomes
chrom_test=['chr1','chr8','chr9','chr2', 'chr3','chr4']


#transform half life from absolute values to relative to the mean
means = []
for row in tissue_hl.itertuples():
    mean = np.nanmean(row[1:50])
    means.append(mean)

tissue_hl.insert(loc=49, column='Mean', value=means)
# disable this loop to predict absolute values, activate it to predict relative to the mean #################### SWITCH ABS/REL

for tissue in tissues:
    tissue_hl[tissue] = tissue_hl[tissue]-tissue_hl['Mean']

tissues.append("Mean")

df_hl_enc = tissue_hl


# encode tissues in a single coloumn
for i in range (len(tissues)):  #49 tissue columns + 1 mean at the end

  df_hl_enc=df_hl_enc.rename(columns={df_hl.columns[i]:i})


#make a row for each hl in tissue/gene combination
df_hl_enc=pd.melt(df_hl_enc, id_vars=df_hl_enc.columns[len(tissues):], value_vars=df_hl_enc.columns[0:len(tissues)], var_name="tissue", value_name="half_life")

#regression needs to be done only once for a given feature / prediction(abs/rel) as its results are stored in a pkl
#lines where path changes might be neccessary are marked with "CHANGE HERE"
lasso_regression()

#load regression results from pkl for further analysis at the end of lasso_regression()
results = pickle.load(open("./pickles/lasso_relative_per_tissue_base.p", "rb"))                                ############## CHANGE HERE

lasso_list = results[0]
df_xpv = results[1]
df_mse = results[2]

# lasso_list[i] ihas xpv at lasso_list[i][0] ,mse at [1], vs at [2] and  coef at [3]

xpv_plot = p9.ggplot(df_xpv, p9.aes('tissue', 'exp_var_sc')) + p9.geom_col() + p9.theme(axis_text_x = p9.element_text(angle = 90))
xpv_plot.save(filename="results/relative/base/xpv_lasso_relative_per_tissue.jpg")                      ############## CHANGE HERE

mse_plot = p9.ggplot(df_mse, p9.aes('tissue', '%root_mean_sq_err')) + p9.geom_col() + p9.theme(axis_text_x = p9.element_text(angle = 90))
mse_plot.save(filename="results/relative/base/mse_lasso_relative_per_tissue.jpg")                        ############## CHANGE HERE

df_coef = pd.DataFrame(columns=lasso_list[0][3].columns)

for i in range(len(tissues)):

    cur_lasso = lasso_list[i][2]
    sns.scatterplot(data=cur_lasso.T, x="pred", y="truth").set(title=tissues[i])

    plt.savefig("results/relative/base/pred_vs_truth/"+tissues[i])                          ############## CHANGE HERE
    plt.clf()
    #built the coefficient dataframe
    df_coef = df_coef.append(lasso_list[i][3])

print()
df_coef = df_coef.replace(to_replace=0.0,value=0.00000000001)
coef_plot = sns.heatmap(df_coef)

plt.savefig("results/relative/heat_coef_relative.png")                                                   ############## CHANGE HERE
plt.clf()

                                                    ############## CHANGE HERE
rsc = df_coef.drop(columns=["TGA","TAA","TAG"])
rsc_plot = sns.heatmap(rsc)
plt.savefig("results/relative/heat_coef_relative_rescale.png")
print()

'''
 
 ################################################################################
# get counts of transcripts per chromosome
chr_hl = {}


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