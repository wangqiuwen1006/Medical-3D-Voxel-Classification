# Medical-3D-Voxel-Classification
This is a classification of lung nodules using pytorch, it is the final project of SJTU-EE228.

## Network Link
Link：https://pan.baidu.com/s/1XFreGXYcs8wS-TGxmKvfwg  
Extraction code：q5wg

## Requirements
- Python 3 (Anaconda 3.5.1 specifically)
- Pytorch==1.4.0

## Code Structure
[acsconv](https://github.com/wangqiuwen1006/Medical-3D-Voxel-Classification/tree/master/acsconv): a converter that turns a 2D network into a 3D network<br />
[densenet](https://github.com/wangqiuwen1006/Medical-3D-Voxel-Classification/blob/master/densenet.py): an implementation of 2D Densenet network, needs to be combined with a converter when using it<br />
[resnet3d](https://github.com/wangqiuwen1006/Medical-3D-Voxel-Classification/blob/master/resnet3d.py): an implementation of 3D Resnet network<br />
[shake_shake](https://github.com/wangqiuwen1006/Medical-3D-Voxel-Classification/blob/master/shake_shake.py): an implementation of Shake_shake network, needs to be combined with a converter when using it<br />
[vgg](https://github.com/wangqiuwen1006/Medical-3D-Voxel-Classification/blob/master/vgg.py): an implementation of 2D vgg network, needs to be combined with a converter when using it<br />
[train_densenet169](https://github.com/wangqiuwen1006/Medical-3D-Voxel-Classification/blob/master/train_densenet169.py): a training program for densenet169 model<br />
[train_mixupresnet50](https://github.com/wangqiuwen1006/Medical-3D-Voxel-Classification/blob/master/train_mixupresnet50.py): a training program for mixup-resnet50 model<br />
[train_mixupresnet50](https://github.com/wangqiuwen1006/Medical-3D-Voxel-Classification/blob/master/train_resnet26.py): a training program for resnet26 model<br />
[train_shake](https://github.com/wangqiuwen1006/Medical-3D-Voxel-Classification/blob/master/train_shake.py): a training program for shake_shake model<br />
[train_vgg](https://github.com/wangqiuwen1006/Medical-3D-Voxel-Classification/blob/master/train_vgg16.py): a training program for vgg model<br />
[test](https://github.com/wangqiuwen1006/Medical-3D-Voxel-Classification/blob/master/vgg.py): load model<br />
## Program Running Instructions
You only need to run the test file to get the prediction of the test data. Please write the test data path before running, and the model and network files are in the same folder
