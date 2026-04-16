<div align="center">

# Applying self-organizing maps to the inverse problem

[![arXiv](https://img.shields.io/badge/arXiv-2604.02958-b31b1b.svg)](https://arxiv.org/abs/2604.02958)

</div>

The inverse problem in particle physics aims to identify a signal hypothesis, given that an excess of events over the prediction is observed. We approach this problem using self-organizing maps (SOMs) for vector-like leptons in a trilepton final state. We compare the performance of SOMs with DNNs. 

Vector-like leptons (VLLs) occur in various BSM models. The model proposes two new particles: L (charged) and N (neutral) (To read more about VLLs one can refer to [ Vectorlike leptons at the Large Hadron Collider. _Phys. Rev. D_, 92(11):115018, 2015](https://doi.org/10.1103/PhysRevD.92.115018). For an example collider search for vector-like leptons, one can read [Inclusive nonresonant multilepton probes of new phenomena at √s=13 TeV. _Phys. Rev. D_, 105:112007, 2022](https://arxiv.org/abs/2202.08676)) The masses of the VLLs are unknown. In our study, we consider the excess observed in a VLL analysis, and aim to pinpoint which VLL mass hypothesis does that excess correspond to. We train the models using VLL masses: 500, 1000, 1500 GeV and test on 750, 1500 GeV as well as standard model processes WZ and TTZ. 

Self-organizing maps are unsupervised clustering ML algorithms. They project the n-dimensional input space onto a two-dimensional grid. We use the 'Minimalistic Implementation of the Self Organizing Maps' or MiniSOM. Procedure to install MiniSOM along with examples can be found [here](https://github.com/JustGlowing/minisom/tree/master).   

There are three main sub-folders in this repository. The `inputs` contains the text files which will be used to train and test both DNNs and SOMs. The `DNN` and `SOM` includes the training and testing python scripts for the respective algorithm, as well as the output generated. 


