# PDP-NCS

This repo implements our paper:

Detian Kong, Yining Ma, Zhiguang Cao, Tianshu Yu and Jianhua Xiao, "[Efficient Neural Collaborative Search for Pickup and Delivery Problems](https://www.researchgate.net/publication/383328030_Efficient_Neural_Collaborative_Search_for_Pickup_and_Delivery_Problems)" in the IEEE Transactions on Pattern Analysis and Machine Intelligence.

## Dependencies
* Python>=3.8
* PyTorch>=1.7
* tensorboard_logger
* tqdm

## Usage

### Training

#### PDTSP examples

21 nodes:
```bash
python run.py --problem pdtsp --graph_size 20 --shared_critic
```

51 nodes:
```bash
python run.py --problem pdtsp --graph_size 50 --shared_critic
```

101 nodes:
```bash
python run.py --problem pdtsp --graph_size 100 --shared_critic
```

#### PDTSP-LIFO examples

21 nodes:
```bash
python run.py --problem pdtspl --graph_size 20 --shared_critic
```

51 nodes:
```bash
python run.py --problem pdtspl --graph_size 50 --shared_critic
```

101 nodes:
```bash
python run.py --problem pdtspl --graph_size 100 --shared_critic
```

If encountered "RuntimeError: CUDA out of memory", please try smaller batch size by adding option ```--batch_size xxx``` (default is 600).

### Inference

Load the model and specify the iteration T for inference (using --val_m for data augments):

```bash
--eval_only 
--load_path '{add model to load here}'
--T_max 3000 
--val_size 2000 
--val_batch_size 200 
--val_dataset '{add dataset here}' 
--val_m 50
```

#### Examples

For inference 2,000 PDTSPL instances with 100 nodes and no data augment (NCS):

```bash
python run.py --eval_only --no_saving --no_tb --problem pdtspl --graph_size 100 --val_m 1 --val_dataset './datasets/pdp_100.pkl' --load_path './pre-trained/ncs/pdtspl_100/epoch-198.pt' --val_size 2000 --val_batch_size 2000 --T_max 3000 --shared_critic
```

For inference 2,000 PDTSPL instances with 100 nodes using the augments (NCS-A):

```bash
python run.py --eval_only --no_saving --no_tb --problem pdtspl --graph_size 100 --val_m 50 --val_dataset './datasets/pdp_100.pkl' --load_path './pre-trained/ncs/pdtspl_100/epoch-198.pt' --val_size 2000 --val_batch_size 200 --T_max 3000 --shared_critic
```

Run ```python run.py -h``` for detailed help on the meaning of each argument.

Datasets for validation and pre-trained model can be found in [release](https://github.com/dtkon/PDP-NCS/releases) of this repos.

## Acknowledgements
The code and the framework are derived from the repos [yining043/PDP-N2S](https://github.com/yining043/PDP-N2S).
