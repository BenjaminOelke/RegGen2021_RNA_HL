

from gagneur_colab.utils.gtf_utils import target_dataframe


gtf_file = '/storage/groups/ml01/workspace/laura.martens/gagneur_colab_data/genome/gencode.v19.annotation.gtf.gz'
fasta_file = None 
feature = '3UTR' # or '5UTR'

results_file = '/home/icb/laura.martens/utr3_gencode19.csv'


df = target_dataframe(gtf_file,
                         feature)

df.to_csv(results_file, sep = '\t')