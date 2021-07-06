#!bin/bash
#a demo for intersecting 120 RBP bed file with the whole-gene-body bed file 
assembly=hg19;
gencode=gencodev19;
bed_dir=/storage/groups/ml01/workspace/laura.martens/gagneur_colab_data/RBP_binding/encode_eclip/$assembly/K562/;
out_dir=/storage/groups/ml01/workspace/laura.martens/gagneur_colab_data/RBP_binding/processed/$assembly/gene;

mkdir -p $out_dir
### use the bedtools to finish intersecting 
ls $bed_dir/*bed* | while read id ; do(nohup bedtools intersect -a "$gencode/humangene_$gencode.bed"  -b $id -F 0.5 -wa -wb | awk '{print $4,$9,$12}' > $out_dir/"$(basename -- $id)".out &);done


### delete blank file 
find $out_dir -name "*" -type f -size 0c | xargs -n 1 rm -f
  
