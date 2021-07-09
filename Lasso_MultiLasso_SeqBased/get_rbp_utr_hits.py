import pandas as pd
import pyranges as pr
import os

# datapaths
utr3_path = './gencode/utr3_gencode19.csv'
utr5_path = './gencode/utr5_gencode19.csv'
h_meta_p = './metadata/metadata_HepG2.csv'
k_meta_p = './metadata/metadata_K562.csv'

# read in half life data to match transcript id's, and UTR annotations for the transcripts
tissue_hl = pd.read_csv('genomic_sequence_plus_features_hl_all_tissues.csv', index_col=0)


utr3 = pd.read_csv(utr3_path, sep = '\t', index_col=0)
utr3 = utr3.loc[utr3.transcript_id.isin(tissue_hl.index)]
utr3_pr = pr.PyRanges(utr3)

utr5 = pd.read_csv(utr5_path, sep = '\t', index_col=0)
utr5 = utr5.loc[utr5.transcript_id.isin(tissue_hl.index)]
utr5_pr = pr.PyRanges(utr5)

# metadata matches RBP with individual cell line files by file accession id
h_meta = pd.read_csv(h_meta_p,index_col=0)
k_meta = pd.read_csv(k_meta_p,index_col=0)

# RBP for each cell line
h_et = set(h_meta["Experiment target"])
k_et = set(k_meta["Experiment target"])

# RBP overlap between celllines
et = h_et.intersection(k_et)

#instantiate feature df
rbp_utr = pd.DataFrame(index=tissue_hl.index,columns=et)
rbp_utr["chromosome"] = tissue_hl["chromosome"]


for t in et:

    #lookup file accession
    hfp = h_meta.loc[h_meta["Experiment target"] == t]["File accession"].reset_index()["File accession"][0]
    kfp = k_meta.loc[k_meta["Experiment target"] == t]["File accession"].reset_index()["File accession"][0]
    #access cell line specific files
    hrbp = pr.read_bed(os.path.join("./RBP-binding/HepG2/", (hfp + ".bed.gz")))
    krbp = pr.read_bed(os.path.join("./RBP-binding/K562/", (kfp + ".bed.gz")))

    #consolidate shared intervalls between cell lines
    rbp_df = pd.concat([hrbp.df, krbp.df])
    rbp = pr.PyRanges(rbp_df)
    rbp = rbp.merge(strand=True)

    # get number of bs for RBP for each transcript in 3" and 5" UTR
    utr3_pr = utr3_pr.count_overlaps(rbp, strandedness='same', overlap_col=t)
    utr5_pr = utr5_pr.count_overlaps(rbp, strandedness='same', overlap_col=t)

    # some transcript id's exist in 5" and 3", we need to account for that and collapse the dataframe so that index_ids have only one unique row
    rbp_utr3 = utr3_pr.df[["transcript_id", t]].set_index("transcript_id")
    rbp_utr5 = utr5_pr.df[["transcript_id", t]].set_index("transcript_id")

    rbp_temp = pd.DataFrame(rbp_utr5[t] + rbp_utr3[t]).fillna(0).reset_index().groupby(by=["transcript_id"])[t].sum()
    rbp_temp = pd.DataFrame(rbp_temp)

    # update feature frame
    rbp_utr[t] = rbp_temp[t]
    print(t)
rbp_utr = rbp_utr.fillna(0)
rbp_utr.to_csv('./results/rbp_utr_gencode_19_binding.csv')


print()




#rbp_utr.to_csv('/results/rbp_utr_gencode_19_binding.csv')