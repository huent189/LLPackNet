# LLPackNet

> The project is the official implementation of our *[BMVC 2020](https://www.bmvc2020-conference.com/assets/papers/0145.pdf) paper, "Towards Fast and Light-Weight Restoration of Dark Images"*<br>  **&mdash; Mohit Lamba, Atul Balaji, Kaushik Mitra**

***A single PDF of the paper and the supplementary is available at [arXiv.org](https://arxiv.org/abs/2011.14133).***

In this work we propose a deep neural network, called `LLPackNet`, that can restore very High Definition `2848×4256` extremely dark night-time images, in just ***3 seconds*** even on a CPU. This is achieved with `2−7× fewer` model parameters, `2−3× lower` memory utilization, `5−20×` speed up and yet maintain a competitive image reconstruction quality compared to the state-of-the-art algorithms.

*Watch the below video for results and overview of LLPackNet.*

[![Watch the project video](https://raw.githubusercontent.com/MohitLamba94/LLPackNet/master/pics/video.png)](https://www.youtube.com/watch?v=nO6pizVH_qM&feature=youtu.be)

<details>
  <summary>Click to read full <i>Abstract</i> !</summary>
  
The ability to capture good quality images in the dark and `near-zero lux` conditions has been a long-standing pursuit of the computer vision community. The seminal work by *Chen et al.* has especially caused renewed interest in this area, resulting in methods that build on top of their work in a bid to improve the reconstruction. However, for practical utility and deployment of low-light enhancement algorithms on edge devices such as embedded systems, surveillance cameras, autonomous robots and smartphones, the solution must respect additional constraints such as limited GPU memory and processing power. With this in mind, we propose a deep neural network architecture that aims to strike a balance between the network latency, memory utilization, model parameters, and reconstruction quality. The key idea is to forbid computations in the High-Resolution (HR) space and limit them to a Low-Resolution (LR) space. However, doing the bulk of computations in the LR space causes artifacts in the restored image. We thus propose `Pack` and `UnPack` operations, which allow us to effectively transit between the HR and
LR spaces without incurring much artifacts in the restored image. <br>


State-of-the-art algorithms on dark image enhancement need to pre-amplify the image before processing it. However, they generally use ground truth information to find the amplification factor even during inference, restricting their applicability for unknown scenes. In contrast, we propose a simple yet effective light-weight mechanism for automatically determining the amplification factor from the input image. We show that we can enhance a full resolution, 2848×4256, extremely dark single-image in the ballpark of 3 seconds even on a CPU. We achieve this with 2−7× fewer model parameters, 2−3× lower memory utilization,
5−20× speed up and yet maintain a competitive image reconstruction quality compared to the state-of-the-art algorithms
 
</details>

# How to use the code?
The ```train.py``` and ```test.py``` files were used for training and testing. Relevant comments have been added to these files for better understanding. You however need to download the [SID dataset](https://github.com/cchen156/Learning-to-See-in-the-Dark) in your PC to execute them. 

The Jupyter Notebooks containing test code for the ablation studies can be also found in the ```ablations``` folder.

We used PyTorch version 1.3.1 with Python 3.7 to conduct the experiment. Along with the commonly used Python libraries such Numpy and Skimage, do install the [Rawpy](https://pypi.org/project/rawpy/) library required to read RAW images.



# Cite us
If you find any information provided here useful please cite us,

<div style="width:600px;overflow:auto;padding-left:50px;">
<pre>
  @inproceedings{lamba2020LLPackNet,
  title={Towards Fast and Light-Weight Restoration of Dark Images},
  author={Lamba, Mohit and Balaji, Atul and Mitra, Kaushik},
  booktitle={British Machine Vision Conference (BMVC)},
  year={2020},
  organization={BMVC}
}
}
</pre>
</div>


