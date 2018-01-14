Seq2Seq Project on Chinese Digits Speech Recognition
=========

b04902004 王佑安

Project Contain
-----------

```
├── data
├── speechdata
├── labels
│   ├── Clean08TR.mlf
│   ├── Clean08TR_sp.mlf
│   └── answer.mlf
├── scripts
│   ├── testing_list.scp
│   └── training_list.scp
├── decoder.pt
├── encoder.pt
├── extract_feat.sh
├── htk_config.cfg
├── model.py
├── readme.txt
├── test.py
├── test.sh
├── train.py
└── utils.py
```


Requirement
------------

```
HTK Tools
python3.6
torch 0.2.0
numpy 1.13.3
editdistance 0.3.1
```

How to Use
--------------
Prepare Data:
	1. cp ```dsp_hw2/speechdata/``` under this directory.
	2. ```bash extrac_feat.sh``` to extract data to mfcc feature under data/ directory.

Train:
```
python3 train.py [-h] [--lr LR] [--epochs EPOCHS] [--batch_size BATCH_SIZE]
				[--hidden_size HIDDEN_SIZE]
optional arguments:
  -h, --help
  --lr LR
  --epochs EPOCHS
  --batch_size BATCH_SIZE
  --hidden_size HIDDEN_SIZE
```
```train.py``` will generate ```encoder.pt``` and ```decoder.pt``` as pretrain model.
		  
Test:
```
python3 test.py [-h] [--batch_size BATCH_SIZE] [--hidden_size HIDDEN_SIZE]
optional arguments:
  -h, --help
  --batch_size BATCH_SIZE
  --hidden_size HIDDEN_SIZE
```
```test.py``` will load pretrained ```encoder.pt``` and ```decoder.pt``` to calculate accuracy on test data.
Note that hidden size should as same as training.
