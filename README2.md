#1 feature exporting
1. to continue creating the file output/experiments/horus-features.setup (TSV) with the id/desc of the data fields (attention: new fields have been added with topic modeling and CNN)
2. to create a centralized function that reads this file and create the final data table with all features .
That is, the features marked with 1 and the derived features (e.g., adding derived features, such as _hasSpecialChar_). This will be the final training/test file for all models.
3. integrate this function/file into the data conversion functions for each algorithm (e.g., DT, CRF, LSTMs, etc..)

#2 re-run experiments
1. run all experiments again and reproduce exact results
2. run all experiments again, comparing performance of new trained text:topic_module and image:cnn modules

#3 image module: CNN
1. CNN models trained by Piyush are not working well yet

#4 web service and demo
1. to release

#5 image CNN
1. instead of object detection, try with object classification in a sparser environment (encode different classes)

#6 reprocessing
1. to optimze the reprocessing job; models should be independent and everything cached into the database. Later, function simply reads the database,
checks if a certain file is processed already and gets the outcomes via SQL command. Everything is too slow now

#7 benchmark integration
1. to integrate the app into GERBIL (benchmark platform) and TANKER (cloud processing)

#8 datasets integration
1. flicker
2. imageNET
3. babelNET
4. Wikipedia

#9 environment deploy
1. docker

-----------------------------------------------------------------------------------------------
- examples/process_input_file.py = generates the data (horus file) for training (horus/core/feature_extraction/main.py)
- experiments/benchmarking.py = training module (part of the feature construction is being processed here, should be integrated (#1))

