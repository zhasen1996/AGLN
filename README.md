# Attention Guided Global Enhancement and Local Refinement Network for Semantic Segmentation

Jiangyun Li, Sen Zha, Chen Chen, Meng Ding, Tianxiang Zhang, and Hong Yu

## Introduction

In this paper, a novel AGLN is proposed to improve the encoder-decoder network for image segmentation. First, a Global Enhancement Method is designed to capture global semantic information from high-level features to complement the deficiency of global contexts in the upsampling process. Then, a Local Refinement Module is built to refine the noisy encoder features in both channel and spatial dimensions before the context fusion. After that, the proposed two methods are integrated into the Context Fusion Blocks, enabling the AGLN to generate semantically consistent segmentation masks on large-scale stuff and accurate boundaries on delicate objects.

AGLN achieves the state-of-the-art result (56.23% mean IOU) on the PASCAL Context dataset.

## Usage



## Citation



## Acknowledge

Thanks [PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding).
