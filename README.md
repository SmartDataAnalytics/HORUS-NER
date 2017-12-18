### Boosting NER
HORUS is meta-algorithm for Named Entity Recognition (NER) based on image processing and multi-level machine learning. It aims at boosting NER task by adding new features to the NER pipeline. Currently supports the identification of classical named-entity types (LOC, PER, ORG). It has been designed (specially) for short-texts.  

#### Easy Setup (Python 2.7 based)

1. Setup the python environment
- setup [Anaconda](https://anaconda.org/)
- conda env create -f horus.v015.yml (tested for OSX) (*)
- source activate horus_27_cv310

2. Setup the framework environment
- setup [SQLite](https://sqlite.org/) database and run [our script](https://github.com/diegoesteves/horus-ner/blob/master/horus0.1.5.db.sql) to create the schema
- get your [Microsoft Bing API Key](https://azure.microsoft.com/en-us/services/cognitive-services/) and [Microsoft Translator API Key](https://datamarket.azure.com/developer/applications/register) to query the Web.
- configure the parameters at the .ini file (copy _horus_dist.ini_ to _~/horus.ini_)

(*) - setup [openCV 3.1](http://www.pyimagesearch.com/2015/06/22/install-opencv-3-0-and-python-2-7-on-ubuntu/): OSx users can benefit from anaconda, which provides a running version of OpenCV 3.1.0 ([see more at #issue 6](https://github.com/dnes85/horus-models/issues/6))

#### Usage 
```python
python main.py --input_text="whitney houston has been honored in nyc" --ds_format=0 --output_format="csv"

python main.py --input_file="sentences.txt" --ds_format=0

python main.py --input_file="ritter_ner.tsv" --ds_format=1 --output_file="metadata" --output_format="json"
```


## Version
- 0.1.0 initial version
- 0.1.1 adding text classification
- 0.1.2 adding map detection
- 0.1.5 paper version
