1 pip install pandas
  
  pip install sklearn

  pip install numpy

  pip install argparse

2 python ForestType.py --input data_CNA.csv.gz --cluster 6 --feature 2000 --output select_feature.csv

3 python RF_Gridsearch.py --input select_feature.csv --quicktrain True --output1 feature_important.csv --output2 fin_result.csv

* Due to upload size limitation, we did not upload mRNA dataset, you can download it from the METABRIC database.