# Attention Guided Global Enhancement and Local Refinement Network for Semantic Segmentation [\[arxiv\]](https://arxiv.org/abs/2204.04363)

Jiangyun Li, Sen Zha, Chen Chen, Meng Ding, Tianxiang Zhang, and Hong Yu

## Introduction

In this paper, a novel AGLN is proposed to improve the encoder-decoder network for image segmentation. First, a Global Enhancement Method is designed to capture global semantic information from high-level features to complement the deficiency of global contexts in the upsampling process. Then, a Local Refinement Module is built to refine the noisy encoder features in both channel and spatial dimensions before the context fusion. After that, the proposed two methods are integrated into the Context Fusion Blocks, enabling the AGLN to generate semantically consistent segmentation masks on large-scale stuff and accurate boundaries on delicate objects.

AGLN achieves the state-of-the-art result (56.23% mean IOU) on the PASCAL Context dataset.

![image](https://github.com/zhasen1996/AGLN/blob/master/img/AGLN.png)

## Usage
Follow [PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding), and our network is in [agln.py](https://github.com/zhasen1996/AGLN/blob/master/encoding/models/sseg/agln.py).

## Citation
```
@misc{https://doi.org/10.48550/arxiv.2204.04363,
  doi = {10.48550/ARXIV.2204.04363},
  
  url = {https://arxiv.org/abs/2204.04363},
  
  author = {Li, Jiangyun and Zha, Sen and Chen, Chen and Ding, Meng and Zhang, Tianxiang and Yu, Hong},
  
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences, I.4.6},
  
  title = {Attention guided global enhancement and local refinement network for semantic segmentation},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {Creative Commons Attribution 4.0 International}
}

```

## Acknowledge

Thanks [PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding).
