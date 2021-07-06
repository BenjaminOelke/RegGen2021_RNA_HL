HLEP_STRING="""

Date : March 20,2021

Author : Laura Martens, Adapted from Ruiyan Hou (ruiyan_hou@163.com) 

This script will describe how to combine all the bed files from 120 RBP to a file.

"""

import pandas as pd
import os.path
from functools import reduce
from tqdm import tqdm

assembly = 'hg19'
gencode = 'gencodev19'
filepath = f'/storage/groups/ml01/workspace/laura.martens/gagneur_colab_data/RBP_binding/processed/{assembly}/gene'

pathDir = os.listdir(filepath)
dfnamels=[]
dfnum=range(1,121)
for allDir,i in zip(pathDir,dfnum):
    child = os.path.join('%s/%s' % (filepath, allDir))
    print(child)
    i = pd.read_csv(child, delimiter=' ', header=None, names = ['gene_id', 'id', 'value'])
    rbp_name = i['id'][0].split('_')[0]
    print(rbp_name)
    i[rbp_name] = i['value'].astype('float')
    i.drop(labels=['value', 'id'], axis=1, inplace=True)
    i=i.groupby(['gene_id']).agg({rbp_name: 'sum'})
    i=pd.DataFrame(i)
    i.reset_index(inplace=True)
    dfnamels.append(i)
print(dfnamels)
df_merged=reduce(lambda left,right:pd.merge(left,right,on=['gene_id'],how='outer'),dfnamels)
df_merged.fillna(0,inplace=True)
df_merged['gene_id']=df_merged['gene_id'].str.split('.',expand=True)[0]
print(df_merged)

lengthdf=pd.read_csv(f'{gencode}/humangene_{gencode}.bed',delimiter='\t')
lengthdf.columns=['chr_id','start_site','stop_site','gene_id', 'transcript_id']
lengthdf['gene_id']=lengthdf['gene_id'].str.split('.',expand=True)[0]
lengthdf['gene_length']=lengthdf['stop_site']-lengthdf['start_site']
lengthdf=lengthdf[['gene_id','gene_length']]
print(lengthdf)

alldf=df_merged.merge(lengthdf,on='gene_id')
print(alldf)

alldf.iloc[:,1:121]=alldf.iloc[:,1:121].div(alldf.gene_length,axis=0)
print(alldf)
alldf.to_csv(f'{assembly}_k562_all_gene_body_RBP.csv')

