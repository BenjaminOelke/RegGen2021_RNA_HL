
from kipoiseq.extractors import FastaStringExtractor, UTRFetcher
import pandas as pd


def target_dataframe(gtf, feature ,fasta = None):
    """
    Extract each interval separetly from gtf file and adding sequence. This can be used to score motifs.

    :param gtf:gtf file 
    :param fasta: fasta file
    :param feature: feature name to extract (i.e 5UTR, 3UTR)
    :return: list of intervals
    """
    interval_fetcher = UTRFetcher(gtf,
                         feature_type=feature,
                         infer_from_cds=True,
                         filter_valid_transcripts=False,
                         filter_biotype=True,
                         filter_tag=False,
                         on_error_warn=True)
    if fasta:
        fasta_fetcher = FastaStringExtractor(fasta)
    else:
        fasta_fetcher = None
    
    # flatten all intervals
    intervals = [item for sublist in interval_fetcher for item in sublist]
    
    def get_dict_from_interval(interval, extractor = None):
        if extractor:
            return {'Chromosome': interval.chrom , 'Start': interval.start , 'End': interval.end, 'Strand': interval.strand ,'Seq': extractor.extract(interval) }
        else:
            return {'Chromosome': interval.chrom , 'Start': interval.start , 'End': interval.end, 'Strand': interval.strand}
    
    interval_dict = [get_dict_from_interval(interval, fasta_fetcher) for interval in intervals]
    
    # convert to df
    interval_df = df = pd.DataFrame(interval_dict)
    
    # merge other information from gtf
    interval_df = interval_df.merge(interval_fetcher.df.reset_index(), on=['Chromosome', 'Start', 'End', 'Strand'], how = 'left')
    
    return interval_df.drop_duplicates()