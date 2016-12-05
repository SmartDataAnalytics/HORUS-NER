## boosting NER
#### HORUS is meta-algorithm for Named Entity Recognition (NER) based on image processing and multi-level machine learning. It aims at boosting NER task by adding <i>apriori</i> information to the pipeline. Currently supports the identification of classical named-entity types (LOC, PER, ORG). It has been designed (specially) short-texts.  

![example](http://dne5.com/whitney_example_peq.png)

## setup
- setup [sqlite](https://sqlite.org/) database and run script to create the schema
- setup [openCV 3](http://docs.opencv.org/)

## usage 
```python
python main.py --input_text="whitney houston has been honored at nyc" --ds_format=0 --output_format="csv"

python main.py --input_file="sentences.txt" --ds_format=0

python main.py --input_file="ritter_ner.tsv" --ds_format=1 --output_file="metadata" --output_format="json"
```
## version
- 0.1 alpha version
