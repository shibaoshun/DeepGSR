# DeepGSR: Deep group-based sparse representation network for solving image inverse problems

*Ke Jiang, Xinya Ji, Baoshun Shi*

## Abstract

In the past few years, group-based sparse representation (GSR) has emerged as a powerful paradigm for image inverse problems by synergizing model-driven interpretability with nonlocal self-similarity priors. Nevertheless, its practical utility is hindered by computationally expensive iterative processes involving nonlocal patch grouping and aggregation. Deep learning (DL) methods can avoid this deficiency by using end-to-end training, but they often lack of model interpretability. To bridge this gap, we propose a novel deep group-based sparse representation framework, termed DeepGSR, which brings the GSR method and the DL approach together. DeepGSR not only circumvents the iterative bottlenecks of conventional GSR but also preserves its intrinsic local sparsity and nonlocal similarity constraints through a learnable parameterization. Specifically, the network is built on a GSR model that enhances both intrinsic local sparsity and nonlocal self-similarity within a unified learnable framework while incorporating the attention mechanism to model complex intra-group relationships, optimizing patch matching and aggregation. To replace the computationally expensive traditional SVD-based rank shrinkage, we introduce a learnable low-rank shrinkage module that integrates both group sparsity and low-rank constraints, offering enhanced interpretability and flexibility. To better exploit frequency-specific structures, the network incorporates a shifting wavelet-domain patch partitioning strategy, which separately models high- and low-frequency components to further enhance the sparsity of both global and local information. Extensive experiments demonstrate that DeepGSR, when applied as a plug-and-play module to various image inverse problems such as image denoising, single-image deraining, metal artifact reduction, sparse-view computed tomography reconstruction, and phase retrieval consistently delivers effective performance, validating the generality of the proposed framework.

<img src="https://github.com/shibaoshun/DeepGSR/blob/main/figs/fig1.jpg" alt="image-20250512165420811"  />

## Installation

This project is based on **PyTorch 1.10.0**. Please make sure you have the correct version installed.

To install the required dependencies, run:

```
pip install -r requirements.txt
```

Make sure you have a CUDA-enabled GPU for training (e.g., NVIDIA 4090).



## Training

### Datasets

You can download the training and testing datasets from the following Baidu Drive link:

📁 `data`

🔗 [https://pan.baidu.com/s/1lQRFUrkaUH7uEDB6iyKq0Q?pwd=2025](https://pan.baidu.com/s/1lQRFUrkaUH7uEDB6iyKq0Q?pwd=2025)

🔑 Password: `2025`

###  Example Training Command

```
python train.py --noiseL 15
```



## Testing

### Prtrained Models

Download the pretrained models from:

📁 `result`

🔗 [https://pan.baidu.com/s/1xbyzy7vOlVyc-vQjKDqcBA?pwd=2025](https://pan.baidu.com/s/1xbyzy7vOlVyc-vQjKDqcBA?pwd=2025)

🔑 Password: `2025`

### Example Testing Command

```
python test.py --noiseL 15
```



## Results

![image-20250512164910204](https://github.com/shibaoshun/DeepGSR/blob/main/figs/fig2.jpg)



## Citation

If you find this code useful, please consider citing our work:

> @article{DeepGSR,
>
>  title   ={DeepGSR: Deep group-based sparse representation network for solving image inverse problems},
>
>  author  = {Ke Jiang, Xinya Ji, Baoshun Shi},
>
>  year   = {2025}
>
> }



## License and Acknowledgement

The codes are based on [DnCNN](https://github.com/cszn/DnCNN). Please also follow their licenses. Thanks for their awesome works. 

We would like to thank all contributors and referenced works for their valuable resources and datasets.



