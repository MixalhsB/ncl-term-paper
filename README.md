## Data

The file _stimuli_erp_Frank_et_al_2015.mat_ contains a part of the freely available dataset by

Frank, S. L., Otten, L. J., Galli, G., & Vigliocco, G. (2015).
The ERP response to the amount of information conveyed by words in sentences.
Brain and language, 140, 1-11.

which can be fully downloaded at:

https://ars.els-cdn.com/content/image/1-s2.0-S0093934X15001182-mmc1.zip

## Requirements

Please install the necessary dependencies by doing:
```
% python3 -m pip install -r requirements.txt  
% python -m nltk.downloader stopwords
```
You also need to have R installed in order to be able to run the packages _rpy2_ and _pymer4_.

## Run analysis

Just type:
```
% python3 run.py
```
