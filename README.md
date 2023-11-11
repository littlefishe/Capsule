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

[IMAGE-100](https://rec.ustc.edu.cn/share/09881460-806a-11ee-bcdf-af03ee393f3c)

## start
```bash
python main.py --dataset IMAGE100 --data_pattern 0
```

## output
client_logs/*
