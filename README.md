# Capsule
Semi-SFL with Clustering Regularization

`Accelarate with torch.multiprocessing`

## requirements
+ torch (>=2.0)
+ torchvision
+ numpy
+ PIL

## dataset & model
|Dataset|Model|
|-|-|
|SVHN | CNN| 
|CIFAR-10 | AlexNet |
|IMAGE-100 | VGG16 |

## download dataset
[SVHN](http://ufldl.stanford.edu/housenumbers/)

[CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar.html)

[IMAGE-100](https://rec.ustc.edu.cn/share/8cb6abd0-806b-11ee-8e09-13f01dde3b5b)

## start
```bash
python main.py --dataset IMAGE100 --data_pattern 0
```

## output
client_logs/*
