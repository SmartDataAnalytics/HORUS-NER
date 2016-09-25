# horus-models
- c++/python classes (java interfaces are not reliable in terms of updates)
- horus machine learning models
- features extraction, clustering, etc. for horus-core algorithm

Overview
- to compute probability for each NER class, based on image processing
- SIFT/SURF + dependency parsing as features
- to find optimal theta for search engine

Related Work
- twitter NER

OpenCV Issues
- there are many issues on java/python bindings with regard to coding SIFT (https://github.com/opencv/opencv/issues/5667 among others) 
- matlibplot does not work on macosx if you don't create a file at /Users/[your-user]/.matplotlib/matplotlibrc and add the following line ```backend: TkAgg```

Datasets
1. PASCAL VOC 2007
2. Caltech-101 - http://www.vision.caltech.edu/Image_Datasets/Caltech101/
3. Flickr
4. 15 scene categories
5. Graz


State of the Art (short text)
- Twitter NER 

Experiments
- DeFacto proof's excerpt
- TwitterNER x HORUS
- Stanford NER x HORUS

Preliminar Results

