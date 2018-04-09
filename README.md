### Boosting NER
HORUS is meta and multi-level framework designed to provide a set of features at word-level to boost natural language frameworks. It's architecure is based on image processing and text classification clustering algorithms and shows to be helpful especially to noisy data, such as microblogs. 

We are currently investigating Named Entity Recognition (NER) as use case. This version supports the identification of classical named-entity types (LOC, PER, ORG). 

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

1. to process an input file (e.g., conll) and generate the horus feature file format
- examples/process_input_file.py

2. to convert a horus input file format to conll
- horus/core/data_conversion.py

3. to run some benchmarks
- experiments/benchmarking.py

## Version
- 0.1.0 initial version
- 0.1.1 adding text classification
- 0.1.2 adding map detection
- 0.1.5 paper version
