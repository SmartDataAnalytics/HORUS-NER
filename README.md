### Boosting NER
HORUS is meta-algorithm for Named Entity Recognition (NER) based on image processing and multi-level machine learning. It aims at boosting NER task by adding new features to the NER pipeline. Currently supports the identification of classical named-entity types (LOC, PER, ORG). It has been designed (specially) for short-texts.  
<p align="center">
<img src=http://dne5.com/whitney_example_peq.png />
</p>

#### Easy Setup (Python 2.7 based)

conda env create -f horus.yml (tested for OSX)
source activate horus_27_cv310

- setup [SQLite](https://sqlite.org/) database and run [our script](https://github.com/dnes85/horus-models/blob/master/horus/cache/database/horus.db.sql) to create the schema
- get your [Microsoft Bing API Key](https://azure.microsoft.com/en-us/services/cognitive-services/) and [Microsoft Translator API Key](https://datamarket.azure.com/developer/applications/register) to query the Web.
- configure the parameters at the .ini file (copy _horus_dist.ini_ to _~/horus.ini_)
- setup [openCV 3.1](http://www.pyimagesearch.com/2015/06/22/install-opencv-3-0-and-python-2-7-on-ubuntu/): OSx users can benefit from anaconda, which provides a running version of OpenCV 3.1.0 ([see more at #issue 6](https://github.com/dnes85/horus-models/issues/6))
- setup [Anaconda](https://anaconda.org/) and import our _requirements.txt_ file (if you're running on linux, please check) 
```python 
conda env create -f environment.yml 
```

#### Config File Parameters

**[path]**
- root_dir: /Users/diegoesteves/Github/named-entity-recognition/horus-ner
- output_path: /Users/diegoesteves/Github/named-entity-recognition/horus-ner/output
- dataset_path: /Users/diegoesteves/Github/named-entity-recognition/horus-ner/data/dataset
- encoder_path: /Users/diegoesteves/Github/named-entity-recognition/horus-ner/horus/resources/models/encoders

**[database]**
- db_path: the path to the *horus.db* file ([download the script here](data/database/horus.db_script_schema.sql))

**[cache]**
- img_folder: /Users/diegoesteves/DropDrive/CloudStation/HORUS_data_paper/cache/img/

**[dataset]**
- ds_ritter: /dataset/Ritter/ritter_ner.tsv
- ds_conll: data/dataset/coNLL2003/nodocstart_coNLL2003.eng.testa
- ds_defacto:

**[search-engine]**
: search engine used to retrieve results / 1 = Microsoft Bing and 2 = Google
- api: 1
- key: please register at the search engine website
- features_text:
- features_img:
- tot_resources: 10

**[translation]**
: please register at https://datamarket.azure.com/developer/applications/register
- microsoft_client_id: 
- microsoft_client_secret:

**[models-param]**
- force_download = True
- location_theta: 5
- distance_theta: 1
- distance_theta_high_bias: True
- safe_interval: 2
- limit_min_loc: -10
- pos_tag_lib = the POS Tagger
    - 1 = nltk pos-tag; 2 = TweetNLP
- pos_tag_lib_type = the Tagset
    - 4 = standard; 5 = universal tagset (that's a trick! keep it 4 or 5)
- kmeans-trees: 5


**[models-cv]**
- horus_loc: /cv/horus_cv_loc.pkl
- horus_org: /horus/org2__svm2.pkl
- horus_org_voc: /horus/dict_002_org2_.pkl
- horus_per: /cv/haarcascade_frontalface_default.xml
- horus_loc_1: /horus/loc_coast_svm2.pkl
- horus_loc_2: /horus/loc_forest_svm2.pkl
- horus_loc_3: /horus/loc_highway_svm2.pkl
- horus_loc_4: /horus/loc_inside_city_svm2.pkl
- horus_loc_5: /horus/loc_mountain_svm2.pkl
- horus_loc_6: /horus/loc_open_country_svm2.pkl
- horus_loc_7: /horus/loc_street_svm2.pkl
- horus_loc_8: /horus/loc_suburb_svm2.pkl
- horus_loc_9: /horus/loc_tall_building_svm2.pkl
- horus_loc_10: /horus/loc_map2_svm2.pkl
- horus_loc_1_voc: /horus/dict_002_loc_coast.pkl
- horus_loc_2_voc: /horus/dict_002_loc_forest.pkl
- horus_loc_3_voc: /horus/dict_002_loc_highway.pkl
- horus_loc_4_voc: /horus/dict_002_loc_inside_city.pkl
- horus_loc_5_voc: /horus/dict_002_loc_mountain.pkl
- horus_loc_6_voc: /horus/dict_002_loc_open_country.pkl
- horus_loc_7_voc: /horus/dict_002_loc_street.pkl
- horus_loc_8_voc: /horus/dict_002_loc_suburb.pkl
- horus_loc_9_voc: /horus/dict_002_loc_tall_building.pkl
- horus_loc_10_voc: /horus/dict_002_loc_map2.pkl

**[models-text]**
- horus_textchecking_1: /horus/resources/models/horus-text/text_classification_LinearSVC.pkl
- horus_textchecking_2: /horus/resources/models/horus-text/text_classification_Passive-Aggressive.pkl
- horus_textchecking_3: /horus/resources/models/horus-text/text_classification_Perceptron.pkl
- horus_textchecking_4: /horus/resources/models/horus-text/text_classification_Ridge Classifier.pkl
- horus_textchecking_5: /horus/resources/models/horus-text/text_classification_SGDClassifier_L1L2.pkl

**[models-horus]**
- horus_final: horus/resources/models/horus-final/randomforest.pkl
- horus_final_encoder: horus/resources/models/horus-final/final_encoder.pkl

**[model-stanford]**
- model_filename_pos: /stanford/2015-04-20/english-bidirectional-distsim.tagger
- path_to_jar_pos: /stanford/2015-04-20/stanford-postagger.jar
- model_filename_ner: /stanford/english.all.3class.distsim.crf.ser.gz
- path_to_jar_ner: /stanford/stanford-ner.jar


#### Distribution
- cd horus-models
- python setup.py sdist

```python
cd dist/
tar -zxvf horus-dist-0.1.tar.gz
python setup.py install --record files.txt
```

## Usage 
```python
python main.py --input_text="whitney houston has been honored in nyc" --ds_format=0 --output_format="csv"

python main.py --input_file="sentences.txt" --ds_format=0

python main.py --input_file="ritter_ner.tsv" --ds_format=1 --output_file="metadata" --output_format="json"
```

## Output
<table>
  <tr>
    <th colspan="3">HORUS MATRIX</th>
  </tr>
  <tr>
    <td>#</td>
    <td>field</td>
    <td>description</td>
  </tr>
  <tr>
    <td>00</td>
    <td>IS_NAMED_ENTITY</td>
    <td>(-1: unknown [test]; 0: no; 1:yes [training])</td>
  </tr>
  <tr>
    <td>01</td>
    <td>ID_SENT</td>
    <td>sentence position</td>
  </tr>
  <tr>
    <td>02</td>
    <td>ID_WORD</td>
    <td>term position</td>
  </tr>
  <tr>
    <td>03</td>
    <td>TOKEN</td>
    <td>word or term (compound)</td>
  </tr>
  <tr>
    <td>04</td>
    <td>POS_UNI</td>
    <td>annotation: universal pos tag</td>
  </tr>
  <tr>
    <td>05</td>
    <td>POS</td>
    <td>pos tag</td>
  </tr>
  <tr>
    <td>06</td>
    <td>NER</td>
    <td>ner tag</td>
  </tr>
  <tr>
    <td>07</td>
    <td>COMPOUND</td>
    <td>compound (0: no 1:yes)</td>
  </tr>
  <tr>
    <td>08</td>
    <td>COMPOUND_SIZE</td>
    <td>size of compound</td>
  </tr>
  <tr>
    <td>09</td>
    <td>ID_TERM_TXT</td>
    <td>id of the table of texts (internal control)</td>
  </tr>
  <tr>
    <td>10</td>
    <td>ID_TERM_IMG</td>
    <td>id of the table of images (internal control)</td>
  </tr>
  <tr>
    <td>11</td>
    <td>TOT_IMG</td>
    <td>total of resources (img) considered (top) = max between tot.retrieved and threshold</td>
  </tr>
  <tr>
    <td>12</td>
    <td>TOT_CV_LOC</td>
    <td>number of resources classified as LOC (computer vision module)</td>
  </tr>
  <tr>
    <td>13</td>
    <td>TOT_CV_ORG</td>
    <td>number of resources classified as ORG (computer vision module)</td>
  </tr>
  <tr>
    <td>14</td>
    <td>TOT_CV_PER</td>
    <td>number of resources classified as PER (computer vision module)</td>
  </tr>
  <tr>
    <td>15</td>
    <td>DIST_CV_I</td>
    <td>distance (subtraction) between 2 max values of (TOT_CV_LOC, TOT_CV_ORG and TOT_CV_PER) (computer vision module)</td>
  </tr>
  <tr>
    <td>16</td>
    <td>PL_CV_I</td>
    <td>sum of all LOC classifiers (computer vision module)</td>
  </tr>
   <tr>
    <td>17</td>
    <td>NR_RESULTS_SE_IMG</td>
    <td>number of images returned from search engine SE for a given term t</td>
  </tr>
  <tr>
    <td>18</td>
    <td>KLASS_PREDICT_CV</td>
    <td>max out of 3cvs</td>
  </tr>
  <tr>
    <td>19</td>
    <td>TOT_RESULTS_TX</td>
    <td>total of resources (snippets of text) considered (top) = max between tot.retrieved and threshold</td>
  </tr>
  <tr>
    <td>20</td>
    <td>TOT_TX_LOC</td>
    <td>number of resources classified as LOC (text classification module)</td>
  </tr>
  <tr>
    <td>21</td>
    <td>TOT_TX_ORG</td>
    <td>number of resources classified as ORG (text classification module)</td>
  </tr>
  <tr>
    <td>22</td>
    <td>TOT_TX_PER</td>
    <td>number of resources classified as PER (text classification module)</td>
  </tr>
  <tr>
    <td>23</td>
    <td>TOT_ERR_TRANS</td>
    <td>number of exceptions raised by the translation module (text classification module)</td>
  </tr>
  <tr>
    <td>24</td>
    <td>DIST_TX_I</td>
    <td>similar to DIST_CV_I (text classification module)</td>
  </tr>
  <tr>
    <td>25</td>
    <td>NR_RESULTS_SE_TX</td>
    <td>number of web sites returned from search engine SE for a given term t</td>
  </tr>
  <tr>
    <td>26</td>
    <td>TX_KLASS</td>
    <td>max out of 3txts</td>
  </tr>
  <tr>
    <td>27</td>
    <td>INDEX_START_TERM</td>
    <td>the position (1st char) of a given term at sentence level</td>
  </tr>
    <tr>
    <td>28</td>
    <td>FEATURE_EXTRA_02</td>
    <td>empty</td>
  </tr>
    <tr>
    <td>29</td>
    <td>FEATURE_EXTRA_03</td>
    <td>empty</td>
  </tr>
    <tr>
    <td>30</td>
    <td>FEATURE_EXTRA_04</td>
    <td>empty</td>
  </tr>
    <tr>
    <td>31</td>
    <td>FEATURE_EXTRA_05</td>
    <td>empty</td>
  </tr>
    <tr>
    <td>32</td>
    <td>FEATURE_EXTRA_06</td>
    <td>empty</td>
  </tr>
    <tr>
    <td>33</td>
    <td>FEATURE_EXTRA_07</td>
    <td>empty</td>
  </tr>
    <tr>
    <td>34</td>
    <td>FEATURE_EXTRA_08</td>
    <td>empty</td>
  </tr>
    <tr>
    <td>35</td>
    <td>FEATURE_EXTRA_09</td>
    <td>empty</td>
  </tr>
  <tr>
    <td>36</td>
    <td>KLASS_01</td>
    <td>CV_KLASS if DIST_CV_I &gt;= self.config.models_distance_theta else TX_KLASS if DIST_TX_I &gt;= self.config.models_distance_theta else 'NONE')</td>
  </tr>
  <tr>
    <td>37</td>
    <td>KLASS_02</td>
    <td>CV_KLASS if DIST_CV_I &gt;= self.config.models_distance_theta+1 else TX_KLASS if DIST_TX_I &gt;= self.config.models_distance_theta+1 else 'NONE')</td>
  </tr>
  <tr>
    <td>38</td>
    <td>KLASS_03</td>
    <td>CV_KLASS if DIST_CV_I &gt;= self.config.models_distance_theta+2 else TX_KLASS if DIST_TX_I &gt;= self.config.models_distance_theta+2 else 'NONE')</td>
  </tr>
  <tr>
    <td>39</td>
    <td>KLASS_04</td>
    <td>Compound Update [based on KLASS_1]</td>
  </tr>
  <tr>
    <td>40</td>
    <td>KLASS_05</td>
    <td>RandomForest Model</td>
  </tr>
  <tr>
    <td>41</td>
    <td>KLASS_06</td>
    <td>empty</td>
    </tr>
  <tr>
    <td>42</td>
    <td>KLASS_07</td>
    <td>empty</td>
  </tr>
  <tr>
    <td>43</td>
    <td>KLASS_08</td>
    <td>empty</td>
  </tr>
  <tr>
    <td>44</td>
    <td>KLASS_09</td>
    <td>empty</td>
  </tr>
  <tr>
    <td>45</td>
    <td>KLASS_10</td>
    <td>empty</td>
  </tr>
  <tr>
    <td>46</td>
    <td>KLASS_11</td>
    <td>empty</td>
  </tr>
  <tr>
    <td>47</td>
    <td>KLASS_12</td>
    <td>empty</td>
  </tr>
  <tr>
    <td>48</td>
    <td>KLASS_13</td>
    <td>empty</td>
  </tr>
  <tr>
    <td>49</td>
    <td>KLASS_14</td>
    <td>empty</td>
  </tr>
  <tr>
    <td>50</td>
    <td>KLASS_15</td>
    <td>empty</td>
  </tr>
  <tr>
    <td>51</td>
    <td>KLASS_REAL</td>
    <td>NER target</td>
  </tr>
</table>

## Version
- 0.1.0 initial version
- 0.1.1 adding text classification
- 0.1.2 adding map detection
