The first part is to network given below:

![asgmnt 2](https://user-images.githubusercontent.com/18267947/48659860-f4131a80-eaab-11e8-9c6c-48eb92a46497.jpg)

The forward & backward propagation are coded manually.

The second part was to remake the network using pytorch and experiment how the network behaves to various changes. The files for part 2 start with naming convention 'Net[file name]'
The results are given below:

[Note: All the results are generated using CPU execution for a max of 10 epoch]

| Model  | Training Accuracy | Validation Accuracy | Training Time for 1 Epoch |
| ------------- | ------------- |------------- | ------------- |
| Raw Code Network  | 87 %  |91% |36.2695 |
| Network using PyTorch  | 94 %  |95 % |83.0270 |

Changing number of hidden units:

| Number of Hidden Nodes  | Training Accuracy | Validation Accuracy | Training Time for 1 Epoch |
| ------------- | ------------- |------------- | ------------- |
| 5 Nodes  | 87 %  |79 % |66.2066 |
| 25 Nodes  | 94 %  |92 % |85.5267 |
| 50 Nodes  | 95 %  |94 % |80.7679 |
| 200 Nodes  | 93 %  |93 % |109.18923 |

Changing the depth of the network, i.e increasing the number of hidden layers

| Network Modification  | Training Accuracy | Validation Accuracy | Training Time for 1 Epoch |
| ------------- | ------------- |------------- | ------------- |
| Removing Hidden Layer  | 84 %  |85 % |66.2066 |
| Adding 1 Hidden Layer (50 Nodes)  | 95 %  |95 % |85.5267 |
| Adding 2 Hidden Layer (50 Nodes + 25 Nodes)  | 85 %  |84 % |80.7679 |
| Adding 2 Hidden Layer (50 Nodes + 150 Nodes)  | 86 %  |86 % |109.18923 |

Changing activation functions

| Activation in order  | Training Accuracy | Validation Accuracy | Training Time for 1 Epoch |
| ------------- | ------------- |------------- | ------------- |
| Relu - Softmax  |95 % |94 % |98.0936 |
| Sigmoid - Relu  | 68 %  |66 % |84.1548 |

Changing Optimizers (Default Parameters)

| Optimizer  | Training Accuracy | Validation Accuracy | Training Time for 1 Epoch |
| ------------- | ------------- |------------- | ------------- |
| Adam  |94 % |94 % |215.4103 |
| RMS Prop  | 95 %  |94 % |184.9818 |

Using Batches

| Batch Size  | Training Accuracy | Validation Accuracy | Training Time for 1 Epoch |
| ------------- | ------------- |------------- | ------------- |
| Batch of 4  |78 % |78 % |35.55023 |
| Batch of 10  | 75 %  |76 % |24.2545 |
| Batch of 100  |26 % |26 % |13.92178 |
