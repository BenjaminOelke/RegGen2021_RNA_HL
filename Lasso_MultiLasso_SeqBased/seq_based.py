# -*- coding: utf-8 -*-
"""seq_based.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1W6JtFYlJEL0yJ6D1yIhrQqRc2w7g7Ar8
"""

!pip install kipoiseq

! pip install tensorflow_addons

import pandas as pd
import numpy as np
from plotnine import *
from tensorflow import keras
import keras.layers as kl
import itertools
import seaborn as sns

! wget -O genomic_sequence_plus_features_hl_all_tissues.csv "https://docs.google.com/uc?export=download&id=1w_HDUL9o66Kmn_LK7aWet0gccVaOAI5n"

class ExplainedVariance(keras.callbacks.Callback):
    def __init__(self, validation_data=(), interval=10):
        super(keras.callbacks.Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = explained_variance_score(self.y_val, y_pred)
            print("interval evaluation - epoch: {:d} - explained variance: {:.6f}".format(epoch, score))

from kipoiseq.transforms.functional import one_hot, fixed_len

tissue_hl = pd.read_csv('genomic_sequence_plus_features_hl_all_tissues.csv', index_col=0)
tissue_hl = tissue_hl.loc[tissue_hl.loc[:, ['3_utr', '5_utr']].dropna().index]
tissues = tissue_hl.columns[:49]
tissue_hl['mean_hl'] = tissue_hl.loc[:, tissues].mean(axis=1)
tissue_hl.loc[:, tissues] = tissue_hl.loc[:, tissues].sub(tissue_hl['mean_hl'], axis=0)
mask_value = -1000
tissue_hl.loc[:, tissues] = tissue_hl.loc[:, tissues].fillna(mask_value)

utr3_seqs = tissue_hl.loc[:,['3_utr']].values.flatten()
utr5_seqs = tissue_hl.loc[:,['5_utr']].values.flatten()
cds_seqs = tissue_hl.loc[:,['cds']].values.flatten()

def pad_sequence(seqs, max_len, anchor='start', value='N'):
  padded_seqs = [fixed_len(seq, max_len, anchor=anchor) for seq in seqs]
  return padded_seqs

max_len_utr3 = 4875
max_len_utr5 = 600
max_len_cds = 3710

fixed_len_seqs_utr3 = pad_sequence(utr3_seqs, max_len_utr3)
fixed_len_seqs_utr5 = pad_sequence(utr5_seqs, max_len_utr5)
fixed_len_seqs_cds = pad_sequence(cds_seqs, max_len_utr5)

one_hot_seqs_utr3 = np.array([one_hot(seq, neutral_value=0) for seq in fixed_len_seqs_utr3])
one_hot_seqs_utr5 = np.array([one_hot(seq, neutral_value=0) for seq in fixed_len_seqs_utr5])
one_hot_seqs_cds = np.array([one_hot(seq, neutral_value=0) for seq in fixed_len_seqs_cds])

codons = [''.join(i) for i in itertools.product(["A","C","T","G"], repeat = 3)]
features = tissue_hl.loc[:, codons].values
chrom_val = ['chr2', 'chr3', 'chr4']
chrom_test = ['chr1', 'chr8', 'chr9']
idx_test = np.where(tissue_hl.chromosome.isin(chrom_test))[0]
idx_val = np.where(tissue_hl.chromosome.isin(chrom_val))[0]
idx_train = np.where(~(tissue_hl.chromosome.isin(chrom_test)| tissue_hl.chromosome.isin(chrom_val)))[0]

import pickle
df_rbp = pd.read_csv('/content/hg19_k562_all_gene_body_RBP.txt', index_col=0)
transcripts = pickle.load(open("/content/t2g.p", "rb"))
transcripts['gene_id'] = transcripts['gene_id'].str.split('.').str[0]
transcripts.set_index('transcript_id', inplace=True)
tissue_hl['gene_id'] = tissue_hl.index.map(transcripts["gene_id"])
df_rbp = tissue_hl.merge(df_rbp, on="gene_id", how="left").drop(columns=tissues).fillna(0.0)
tissue_hl = tissue_hl.drop(columns=("gene_id")) 
df_rbp = df_rbp.drop(columns=['cds', '3_utr', '5_utr', 'all'])

df_mirna = pd.read_csv("/content/Number_RNA_family.csv",index_col=0)
df_mirna = df_mirna.set_index("id")
df_mirna = df_mirna.loc[tissue_hl.index].to_numpy()

df_rbp=df_rbp.drop(columns=['gene_length','mean_hl','gene_id','chromosome',"gc_content_3_utr",  "gc_content_5_utr",  "gc_content_cds","log_3_utr_length",  "log_5_utr_length",  "log_cds_length","TGA","TAA","TAG",'AAA', 'AAC','AAG', 'AAT', 'ACA', 'ACC', 'ACG', 'ACT', 'AGA', 'AGC', 'AGG', 'AGT','ATA', 'ATC', 'ATG', 'ATT', 'CAA', 'CAC', 'CAG', 'CAT', 'CCA', 'CCC','CCG', 'CCT', 'CGA', 'CGC', 'CGG', 'CGT', 'CTA', 'CTC', 'CTG', 'CTT','GAA', 'GAC', 'GAG', 'GAT', 'GCA', 'GCC', 'GCG', 'GCT', 'GGA', 'GGC',
            'GGG', 'GGT', 'GTA', 'GTC', 'GTG', 'GTT', 'TAC', 'TAT','TCA', 'TCC', 'TCG', 'TCT', 'TGC', 'TGG', 'TGT', 'TTA', 'TTC','TTG', 'TTT']).to_numpy()

def train_test_split(array, idx_train, idx_val, idx_test):
  return array[idx_train], array[idx_val], array[idx_test]

utr3_train, utr3_val, utr3_test = train_test_split(one_hot_seqs_utr3, idx_train, idx_val, idx_test)
utr5_train, utr5_val, utr5_test = train_test_split(one_hot_seqs_utr5, idx_train, idx_val, idx_test)
cds_train, cds_val, cds_test = train_test_split(one_hot_seqs_cds, idx_train, idx_val, idx_test)
codons_train, codons_val, codons_test = train_test_split(features, idx_train, idx_val, idx_test)
rbp_train,rbp_test,rbp_val = train_test_split(df_rbp,idx_train,idx_test,idx_val)
mirna_train,mirna_test,mirna_val = train_test_split(df_mirna,idx_train,idx_test,idx_val)
y_vars = list(tissues) + ['mean_hl']
y_train, y_val, y_test = train_test_split(tissue_hl.loc[:, y_vars].values, idx_train, idx_val, idx_test)

from keras import backend as K

def function_masked_mse(y_true, y_pred):
        mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())
        masked_summed_error = K.sum(K.square(mask * (y_true - y_pred)), axis=1)
        smooth=0
        masked_mean_squared_error = masked_summed_error / (K.sum(mask, axis=1) + smooth)

        return masked_mean_squared_error

from numpy.random import seed
seed(1)
from tensorflow.random import set_seed
set_seed(2)
from keras.callbacks import EarlyStopping, History
from keras.models import Model
from sklearn.metrics import explained_variance_score
import tensorflow_addons as tfa

# Codon network
inputs_codons = kl.Input(features.shape[1:])

x = kl.Dense(units=64)(inputs_codons)
x = kl.ReLU()(x)
n_hidden = 2
for layer in range(n_hidden):
  x = kl.Dense(units=64)(x)
  x = kl.Dropout(0.2)(x)
  x = kl.ReLU()(x)
output_codons = x

#mirna network
'''
inputs_mirna = kl.Input(df_mirna.shape[1:])

x4 = kl.Dense(units=120)(inputs_mirna)
x4 = kl.ReLU()(x4)
n_hidden = 2
for layer in range(n_hidden):
  x4 = kl.Dense(units=120)(x4)
  x4 = kl.Dropout(0.2)(x4)
  x4 = kl.ReLU()(x4)
output_mirna = x4
'''
#rbp network
inputs_rbp = kl.Input(df_rbp.shape[1:])

x3 = kl.Dense(units=120)(inputs_rbp)
x3 = kl.ReLU()(x3)
n_hidden = 2
for layer in range(n_hidden):
  x3 = kl.Dense(units=120)(x3)
  x3 = kl.Dropout(0.2)(x3)
  x3 = kl.ReLU()(x3)
output_rbp = x3

# 3' UTR network
input_utr3 = kl.Input((one_hot_seqs_utr3.shape[1:]))
x1 = kl.Conv1D(32, kernel_size = 10, padding='same')(input_utr3)
x1 = kl.ReLU()(x1)
x1 = kl.Conv1D(32, kernel_size = 10, padding='same', dilation_rate=2)(x1)
x1 = kl.ReLU()(x1)
x1 = kl.Conv1D(32, kernel_size = 10, padding='same', dilation_rate=4)(x1)
x1 = kl.ReLU()(x1)
x1 = kl.Conv1D(32, kernel_size = 10, padding='same', dilation_rate=16)(x1)
x1 = kl.ReLU()(x1)
x1 = kl.Conv1D(32, kernel_size = 10, padding='same', dilation_rate=16^2)(x1)
x1 = kl.ReLU()(x1)
x1 = kl.GlobalMaxPool1D()(x1)

# 5' UTR network
input_utr5 = kl.Input((one_hot_seqs_utr5.shape[1:]))
x2 = kl.Conv1D(32, kernel_size = 10, padding='same')(input_utr5)
x2 = kl.ReLU()(x2)
x2 = kl.Conv1D(32, kernel_size = 10, padding='same', dilation_rate=2)(x2)
x2 = kl.ReLU()(x2)
x2 = kl.Conv1D(32, kernel_size = 10, padding='same', dilation_rate=4)(x2)
x2 = kl.ReLU()(x2)
x2 = kl.Conv1D(32, kernel_size = 10, padding='same', dilation_rate=16)(x2)
x2 = kl.ReLU()(x2)
x2 = kl.Conv1D(32, kernel_size = 10, padding='same', dilation_rate=16^2)(x2)
x2 = kl.ReLU()(x2)
x2 = kl.GlobalMaxPool1D()(x2)

# cds network
input_cds = kl.Input((one_hot_seqs_cds.shape[1:]))
x5 = kl.Conv1D(32, kernel_size = 10, padding='same')(input_cds)
x5 = kl.ReLU()(x5)
x5 = kl.Conv1D(32, kernel_size = 10, padding='same', dilation_rate=2)(x5)
x5 = kl.ReLU()(x5)
x5 = kl.Conv1D(32, kernel_size = 10, padding='same', dilation_rate=4)(x5)
x5 = kl.ReLU()(x5)
x5 = kl.Conv1D(32, kernel_size = 10, padding='same', dilation_rate=16)(x5)
x5 = kl.ReLU()(x5)
x5 = kl.Conv1D(32, kernel_size = 10, padding='same', dilation_rate=16^2)(x5)
x5 = kl.ReLU()(x5)
x5 = kl.GlobalMaxPool1D()(x5)


# Combine UTR networks for shared layers

x = kl.concatenate([x1,x5,x2, output_codons,output_rbp])
x = kl.Dense(units = 128)(x)
#x = kl.BatchNormalization()(x)
x = kl.ReLU()(x)
output = kl.Dense(units=50)(x)
model = Model(inputs=[input_utr3,input_cds,input_utr5, inputs_codons,inputs_rbp], outputs=output)

model.compile(optimizer=keras.optimizers.Adam(lr = 0.001), loss=function_masked_mse)



tissue_m = model.fit([utr3_train,cds_train,utr5_train, codons_train,rbp_train], 
                    y_train, 
                    validation_data=([utr3_val,cds_val,utr5_val, codons_val,rbp_val], y_val),
                    callbacks=[EarlyStopping(patience=10, restore_best_weights=True),   
                               History()],
                    batch_size=10,  
                    epochs=200)

import matplotlib.pyplot as plt
def plot_loss(history):
    fig, ax = plt.subplots(figsize = (5,5))
    ax.plot(history['loss'][1:])
    ax.plot(history['val_loss'][1:])
    plt.xlabel('epoch')
    plt.ylabel('mean squared error')
plot_loss(tissue_m.history)

def plot_prediction(y_true, x, model):
    fig, ax = plt.subplots(figsize = (5,5))
    ypred = model.predict(x)
    ax.plot(ypred, y_true, '.');
    ax.text(0.95, 0.95, f'ev: {explained_variance_score(y_true, ypred)}',
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes, fontsize=12)
    plt.xlabel('predicted half life')
    plt.ylabel('measured half life')

import plotnine as p9

# This function inputs 2 dataframes with tasks as columns and 
# mRNAs as rows. One data frame contains the true (measured) values
# and the other the predicted ones.
# The output is a dataframe with 
# 2 columns: task and explained variance score

def get_scores(true_df, pred_df):

  exp_var_scores = []
  spearman_scores = []

  for y_var in y_vars:
    non_na_idxs = true_df[true_df[y_var]!= mask_value].index
    exp_var_scores.append(explained_variance_score(true_df.loc[non_na_idxs, y_var].values, pred_df.loc[non_na_idxs, y_var].values))

  scores_df = pd.DataFrame({'task':y_vars, 'exp_var_score': exp_var_scores})

  return scores_df

preds_val = model.predict([utr3_val,cds_val,utr5_val, codons_val,rbp_val])

preds_val_df = pd.DataFrame(preds_val, columns=y_vars, 
                                     index=tissue_hl.iloc[idx_val].index)

true_val_df = tissue_hl.iloc[idx_val].loc[:, y_vars]


# Get the score
val_scores_df = get_scores(true_val_df, preds_val_df)

# Plot scores per task
p9.ggplot(val_scores_df, p9.aes('task', 'exp_var_score')) + p9.geom_col() + p9.theme(axis_text_x = p9.element_text(angle = 90))

val_scores_df.to_csv("seq_based_base_rbp_val_xpv_with_cds.csv")
preds_val_df.to_csv("seq_based_base_rbp_val_pred_with_cds.csv")
true_val_df.to_csv("seq_based_base_rbp_val_true_with_cds.csv")

preds_test = model.predict([utr3_test,cds_test ,utr5_test, codons_test,rbp_test])
preds_test_df = pd.DataFrame(preds_test, columns=y_vars, 
                                     index=tissue_hl.iloc[idx_test].index)

true_test_df = tissue_hl.iloc[idx_test].loc[:, y_vars]
# Get the scores
test_scores_df = get_scores(true_test_df, preds_test_df)

# Plot scores per task
p9.ggplot(test_scores_df, p9.aes('task', 'exp_var_score')) + p9.geom_col() + p9.theme(axis_text_x = p9.element_text(angle = 90))

test_scores_df.to_csv("seq_based_base_rbp_test_xpv_with_cds.csv")
preds_test_df.to_csv("seq_based_base_rbp_test_pred_with_cds.csv")
true_test_df.to_csv("seq_based_base_rbp_test_true_with_cds.csv")