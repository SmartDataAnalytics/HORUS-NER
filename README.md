## HORUS-NER: A Named Entity Recognition Framework for Noisy Data


### Milestones

- HORUS 1.0 (SIFT binary + CRF)
- HORUS 2.0 (B-LSTM + CRF)
- HORUS 3.0 (+attention)
- HORUS 4.0: (+linking) = OCTOPUS

### Data log

HORUS_DB


Last update: 09 Sep 2018

| Table        | Rows           |
| ------------- |:-------------:| 
| HORUS_SEARCH_RESULT_IMG      | 1.170,587 | 
| HORUS_SEARCH_RESULT_TEXT      | 2.119,883      | 
| HORUS_SEARCH_TYPES | 2      | 
| HORUS_SENTENCES | 22.802      | 
| HORUS_TERM | 23.689      | 
| HORUS_TERM_SEARCH | 431.950      |  

### Features

1. To run NER (standard 3-MUC labels)
2. To open for training (training-template-files)
    - for each class k, a set of seed _news_ and _images_ to generalize the model (distant supervision like.). Example: recognize proteins.
    
## How it works?

In the following we present the steps required to generated the dump data to train your own NER model using our features.

#### Pre-processing, Transformations and Feature Extraction

Follow the next steps to initialize the framework, setup cache and extract the features. 
By default, the data dumps will be saved inside the ```resources/``` folder. 
You can also define a different path to the dump files in the ```horus.ini``` config file 
(see more below...).

```
[resources]
# leave it blank to use the project root path
resources_dir: /set/another/path/for/resources
images_dir: /set/another/path/for/cached/images
``` 

```scripts/```

**Step 1:** Setup 

- configure the system parameters in ```horus_dist.ini```
- rename the file to ```horus.ini```
- ```bash ./00_initialize.sh``` to set up necessary configurations. 

The following parameters should be confirmed:
```
[path]
resources_dir: /path/to/resources/folder/
images_dir: /path/to/images/folder/
```

**Step 2**: Generate the Part-of-speech (POS) Encoders: 

````python 01_encoders.py````

```
- All annotators
    * _encoder_pos.pkl (all annotators)
- NLTK
    * _encoder_nltk.pkl
    * _encoder_nltk_universal.pkl
- Stanford
    * _encoder_stanford.pkl
    * _encoder_stanford_universal.pkl
- TweetNLP
    * _encoder_tweetnlp.pkl
    * _encoder_tweetnlp_universal.pkl   
```

**Step 3:** Perform tokenization (to add POS) and class alignment (CoNLL x Tokenizer)

```
python 02_conll2horus.py
```

In this step we generate the metadata for each dataset:
    
```
{
	"dataset": "",
	"sentences": {
		"idx": 1,
		"text": "",
		"language": "",
		"tot_tokens": 3,
		"tokens": {
			"idx": 0,
			"text": "",
			"idx_start": 0,
			"idx_end": 3,
			"pos_tagger": "NN",
			"gold_ner": "",
			"gold_uri": "",
			"language": "",
			"lexical": {
				"lower": "",
				"lemma": "",
				"steam": "",
				"is_len_token_eq_1": "",
				"has_special_char": "",
				"has_first_upper": "",
				"is_upper": "",
				"is_title": "",
				"is_digit": "",
				"has_stop": "",
				"is_len_token_less_3": "",
				"has_minus_in_token": "",
				"shape_id": "",
				"brown_320": "",
				"brown_640": "",
				"brown_1000": ""
			},
			"images": {
				"id": "",
				"predictions": {
					"cls_1_prediction": "",
					"cls_2_prediction": "",
					"cls_3_prediction": "",
					"cls_4_prediction": "",
					"cls_5_prediction": ""
				}
			},
			"news": {
				"id": "",
				"predictions": {
					"cls_1_prediction": "",
					"cls_2_prediction": "",
					"cls_3_prediction": "",
					"cls_4_prediction": "",
					"cls_5_prediction": ""
				}
			}
		}
	}
}
```

**Step 4:** Initialize the cache: News and Images

This steps creates the cache we need to later extract the ```textual``` and ```visual features```.

**Step 5:** Feature Extraction

Here we perform feature extraction. This updates the metadata file with ```lexical```, ```word clusters```, ```visual``` and ```images```.
The metadata file (```*.horus.json```) can now be used to train your own NER model. See the folder ```/notebooks``` for
examples and experiments.

 


   
   

    