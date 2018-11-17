This lab aims to make a custom layer compatible to the auto grad function of pytorch.

The layer implemented is MaxOut Layer :
https://arxiv.org/pdf/1302.4389.pdf

The iPythonNotebooks in order of X, Y & Z represent a network with a Maxout layer, a vanilla network (for comparison) & Maxout layer with drop outs.

The dataset used here is CIFAR10. The checkFile.py file ensures the file auto-downloads if not available.