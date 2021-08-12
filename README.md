# HiggsHadron-DNN

## Introduction
This is a DNN based package for the classification of Higgs boson hadronic decay.
Currently, the package is used to perform the 6 and 7 classes calssification. So there are two
bash scripts, DNN_multi_class_6.sh and DNN_multi_class_7.sh.
And the package is still developing.

## Install
This project uses DNN. To begin with
```
git clone git@github.com:Wujinfei/HiggsHadron-DNN.git
```

Then it is based on cvmfs,
```sh
source /cvmfs/sft.cern.ch/lcg/views/LCG_96python3/x86_64-centos7-gcc8-opt/setup.sh
```

## How to run
./DNN_multi_class_6.sh config_6.yaml
