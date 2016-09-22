# horus-models
- c++/python classes (java interfaces are not reliable in terms of updates)
- horus machine learning models
- features extraction, clustering, etc. for horus-core algorithms

Overview
- to compute probability for each NER class, based on image processing
- SIFT/SURF + dependency parsing as features
- to find optimal theta for search engine

Related Work
- twitter NER

OpenCV Issues
- there are many issues on java/python bindings with regard to coding SIFT (https://github.com/opencv/opencv/issues/5667 among others) 
- matlibplot does not work on macosx if you don't create a file at /Users/[your-user]/.matplotlib/matplotlibrc and add the following line ```backend: TkAgg```
