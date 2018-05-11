### Experiment configuration

- ID: 023-prod
- Description: Enhancing HORUS through NNs based-architectures
- Name: Esteves, Diego
- Contact: name surname at gmail dot com
- Date: 11 May 2018
- Metadata file: horus_023.mex
- Github commit: b746108

#### Datasets

- Ritter
- WNUT-15
- WNUT-16
- WNUT-17

#### Scripts

- ```src/core/feature_extraction/features.py```

    HORUS feature extractor

    ```python
    python features.py --input_dataset_file="ner.txt" --label="ritter" --load_sift=1 --load_tfidf=1 --load_cnn=1 --load_topic_modeling=1
    ```

- ```src/classifiers/benchmarking.py```

    Benchmarks different classifiers with and without the HORUS feature's list.

    ```python
    python benchmarking.py --input_dataset_file="ritter.horus"
    ```

- ```src/core/feature_extraction/horus_to_conll.py```

    Converts a horus metadata file to CoNLL format (sentence spaces and no extra metadata information). Especially useful for evaluating the performance of the framework in 3rd party scripts (**after running the trained models**).

    ```python
    python horus_to_conll.py --exp='EXP_000' --ds='ritter.horus' --f=[3,4,5]
    ```

    exports the file **ritter.horus.conll** with the [features' index](https://github.com/SmartDataAnalytics/horus-ner/blob/master/horus_output.html) 3, 4 and 5

#### FAIR Data Principles
(Wilkinson et al., 2016)

- In order to achieve *full reproducibility* and delivery high-quality metadata, all parameters of the experiment (including hyper-parameters and several "must-information") as well as experiment outcomes are exported to a machine learning graph-based file format dubbed [MEX vocabulary](https://github.com/METArchive/mex-vocabulary) (Esteves et al., 2015).

#### Questions?

- Please open an [issue](https://github.com/SmartDataAnalytics/horus-ner/issues) and we'll be very much happy to help!