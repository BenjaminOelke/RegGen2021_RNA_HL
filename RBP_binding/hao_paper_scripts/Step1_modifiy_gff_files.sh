#!bin/bash
# a demo for downloading gff file and selecting the gene information

### 1. download human genome annotation file
wget ftp://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_24/gencode.v24.annotation.gff3.gz
#wget ftp://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_19/gencode.v19.annotation.gff3.gz

#gzip -d gencode.v19.annotation.gff3.gz
gzip -d gencode.v24.annotation.gff3.gz

### 2. select gene information from gff annotation file
# awk '$3=="gene"' gencode.v19.annotation.gff3 > gene_gencode.v19.annotation.gff3
# awk '$3=="transcript"' gencode.v19.annotation.gff3 > transcript_gencode.v19.annotation.gff3

awk '$3=="gene"' gencode.v24.annotation.gff3 > gene_gencode.v24.annotation.gff3
awk '$3=="transcript"' gencode.v24.annotation.gff3 > transcript_gencode.v24.annotation.gff3

