==========================================================
HORUS: Named Entity Recognition Algorithm
==========================================================

HORUS is a Named Entity Recognition Algorithm specifically
designed for short-text, i.e., microblogs and other noisy
datasets existing on the web, e.g.: social media, some web-
sites, blogs and etc..

It is a simplistic approach based on multi-level machine
learning combined with computer vision techniques that aims
at provide metadata for each token of an input sequence.
Example:

    #!/usr/bin/env python
    from horus.components.core import Core

    horus = Core(False, 5)
    x = horus.annotate("j bond was born in glencoe")
    print x

more info at: https://github.com/dnes85/components-models

# Author: Esteves <diegoesteves@gmail.com>
# Version: 1.5
# Version Label: HORUS_NER_2016_1.5
# License: BSD 3 clause