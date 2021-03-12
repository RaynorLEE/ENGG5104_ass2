# ENGG5104_ass2
ENGG5104 Image Processing and CV assignment2

For Q2, please comment line 65, 66, 67, 81, 102 and uncomment line 79, 100 in train.py

For Q3, please comment line 81, 102 and uncomment line 65, 66, 67, 79, 100 in train.py

For Q4, please comment line 81, 100, and uncomment line 65, 66, 67, 79, 102 in train.py 

For Q5, please comment line 79, 100, and uncomment line 65, 66, 67, 81, 102 in train.py

line 65:	Padding(padding=4),
line 66:	RandomCrop(size=32),
line 67:	RandomFlip(),

line 79:	model=alexnet(num_classes=args.num_classes)
line 81:	model = ResNet()
line 100:	ce_train = CrossEntropyLoss()
line 102:	CrossEntropyLoss(cls_count=class_count, dataset_size=dataset_size)

./code contains all the source code
./result contains sample experiment output and model pth file

Reference Accuracy:
Q2:	53%
Q3:	67%
Q4:	71%
Q5:	78%
