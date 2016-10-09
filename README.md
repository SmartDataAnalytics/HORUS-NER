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

Datasets (Computer Vision)
1. PASCAL VOC - http://host.robots.ox.ac.uk/pascal/VOC/
2. Caltech-101 - http://www.vision.caltech.edu/Image_Datasets/Caltech101/
3. Flickr
4. 15 scene categories
5. Graz

Datasets (NER)
1. DeFacto's proofs/evidences
2. Ritter et al., 2011 (ANNIE NER pipelines x Twitter specific approach)
    System         Precision Recall  F1
    --------------- Newswire ------------- 
    ANNIE            78%      74%   77%
    Stanford          -        -    89%
    -------------- Microblog --------------
    ANNIE            47%      83%   60%
    TwitIE           77%      83%   80%
    Stanford         59%      32%   41%
    Stanford-twitter 54%      45%   49%
    Ritter           73%      49%   59%

State of the Art (short text)
- Twitter NER 

Experiments
- DeFacto proof's excerpt
- TwitterNER x HORUS
- Stanford NER x HORUS

Preliminar Results

