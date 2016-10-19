#ResNet-ImageNet-for-caffe

This is a generator used to get ResNet/Pre-activation ResNet/Inception-ResNet-v2 prototxt for caffe to train ImageNet.

#Usage:
python resnet-imagenet.py solver.prototxt train_val.prototxt --layer_num N (N is in [18, 34, 50, 101, 152])
python pre-resnet-imagenet.py solver.prototxt train_val.prototxt --layer_num N (N is in [50, 101, 152, 200])
python inception-resnet-v2.py solver.prototxt train_val.prototxt

You need to change the path of the training data and the test data(LMDB) in the data layers.
The size of images in the first and the second is 224x224, and in the third is 328x328.


Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun: Deep Residual Learning for Image Recognition, 2015, https://arxiv.org/abs/1512.03385
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun: Identity Mappings in Deep Residual Networks, 2016, https://arxiv.org/abs/1603.05027
[3] Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi: Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning, 2016, https://arxiv.org/abs/1602.07261
