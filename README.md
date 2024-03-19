<div align="center">

# Dysen-VDM: Empowering Dynamics-aware Text-to-Video Diffusion with LLMs

[Hao Fei](http://haofei.vip/), [Shengqiong Wu](https://chocowu.github.io/), 
[Wei Ji](https://jiwei0523.github.io/), 
[Hanwang Zhang](https://mreallab.github.io/people.html), 
and [Tat-Seng Chua](https://chuatatseng.com/)


</div>
<div align="justify">


-----

<a href='http://haofei.vip/Dysen-VDM/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
<a href='https://huggingface.co/spaces/xxxx/Dysen-VDM'><img src='https://img.shields.io/badge/Demo-Page-purple'></a> 
<a href='https://arxiv.org/pdf/2308.13812'><img src='https://img.shields.io/badge/Paper-PDF-orange'></a> 
![License](https://img.shields.io/badge/License-BSD-blue.svg)


This is the repository that contains the source code for the CVPR 2024 paper of **Dysen-VDM**.



----------------------------------



# Framework architecture

<p align="center" width="100%">
<a target="_blank"><img src="Figures/framework.png" alt="Dysen-VDM" style="width: 100%; min-width: 200px; display: block; margin: auto;"></a>
</p>




<p align="center" width="100%">
<a target="_blank"><img src="Figures/imagination.png" alt="Dysen-VDM" style="width: 100%; min-width: 200px; display: block; margin: auto;"></a>
</p>




----------------------------------

## ⚙️ Setting environments

### Install Environment via Anaconda
```bash
conda create -n dysen_vdm python=3.8.5
conda activate dysen_vdm
pip install -r requirements.txt
```

### Download Datasets

Put all the data at `dataset` fold.

1) Pre-training corpus
   - WebVid
     - WebVid is a large-scale dataset of videos with textual descriptions, where the videos are diverse and rich in their content. 
     - There are 10.7M video-caption pairs, where we only use 3M text-video pairs for the pre-training of VDM.
     - The dataset can be downloaded from the [official website](https://m-bain.github.io/webvid-dataset/), and save them in the `dataset/webvid`.

2) Text-to-video in-domain data
   - UCF-101
      - Composed of diverse human actions, which contains 101 classes where each class label denotes a specific movement label.
      - The dataset can be downloaded from the [official website](https://www.crcv.ucf.edu/data/UCF101.php), and save them in the `dataset/ucf101`.

   - MSR-VTT
      - MSR-VTT (Microsoft Research Video to Text) is a large-scale text-video pair 715 dataset. It consists of 10,000 video clips from 20 categories, and each video clip is annotated with 20
      716 English sentences by Amazon Mechanical Turks.
      - The dataset can be downloaded from the [official website](https://www.microsoft.com/en-us/research/publication/msr-vtt-a-large-video-description-dataset-for-bridging-video-and-language/), and save them in the `dataset/msrvtt`.

   - ActivityNet
      - Each video in ActivityNet connects to the descriptions with multiple actions (at least 3 actions), allowing to describe multiple complex events that occur.
      - The dataset can be found in the [official website](http://activity-net.org/download.html), and save them in the `dataset/activityNet`.



---
## 💫 Pre-training Dysen-VDM

We first pre-train the Dysen-VDM system.
The pre-training process is with the `dataset/WebVid` text-video pair data.


### Step 1: Pre-train the video autoencoder of VDM



```
bash shellscripts/train_vdm_autoencoder.sh 
```
- Properly set up  `PROJ_ROOT`, `DATADIR`, `EXPERIMENT_NAME` and `CONFIG`, where
`EXPERIMENT_NAME` = `webvid`.


### Step 2: Pre-train the backbone VDM for text-conditioned video generation
```
bash shellscripts/run_train_vdm.sh
```
- Properly set up  `PROJ_ROOT`, `DATADIR`, `AEPATH`, `EXPERIMENT_NAME` and `CONFIG`, where
`EXPERIMENT_NAME` = `webvid`.

This step uses gold DSG of video for the updating of recurrent graph Transformer in 3D-UNet.
parse the DSG annotations in advance with the tools in `dysen/DSG`


### Step 3: (Post-)Train the overall Dysen-VDM with dynamic scene managing

```
bash shellscripts/run_train_dysen_vdm.sh
```
- properly set up `EXPERIMENT_NAME`, `RESUME`, `DATADIR`, `CKPT_PATH` and `VDM_MODEL`, where
`EXPERIMENT_NAME` = `webvid`.

- The in-context learning (ICL) process within dysen is optimized with reinforcement learning (RL). 
If using RL for the `Imagination Rationality` optimization, gold DSG of video is needed.
parse the DSG annotations in advance with the tools in `dysen/DSG`.


---
## 🧩 Fine-tuning Dysen-VDM on in-domain data

We further update Dysen-VDM on the in-domain training set:

```
bash shellscripts/run_train_dysen_vdm.sh
```
- Properly set up `EXPERIMENT_NAME`, `RESUME`, `DATADIR`, `CKPT_PATH` and `VDM_MODEL`, where
`EXPERIMENT_NAME` = `activityNet` | `msrvtt` | `ucf101`.



---
## 💫 Evaluating 

Measuring the performances of Dysen-VDM on datasets `dataset`:

```
bash shellscripts/run_eval_dysen_vdm.sh
```
- Properly set up `DATACONFIG`, `PREDCITPATH`, `GOLDPATH`, `EXPERIMENT_NAME`, and `RESDIR`.



---
## 💫 Inference

Text-to-video generation with well-trained Dysen-VDM:

```
bash shellscripts/run_sample_vdm_text2video.sh
```



----------------------------------

## Contact

For any questions or feedback, feel free to contact [Hao Fei](mailto:haofei37@nus.edu.sg).


## Citation

If you find Dysen-VDM useful in your research or applications, please kindly cite:
```
@inproceedings{fei2024dysen,
  title={Dysen-VDM: Empowering Dynamics-aware Text-to-Video Diffusion with LLMs},
  author={Hao Fei, Shengqiong Wu, Wei Ji, Hanwang Zhang, Tat-Seng Chua},
  booktitle={Proceedings of the CVPR},
  pages={961--970},
  year={2024}
}
```




## License Notices
This repository is under [BSD 3-Clause License](LICENSE.txt).
Dysen-VDM is a research project intended for non-commercial use only. 
One must NOT use the code of Dysen-VDM for any illegal, harmful, violent, racist, or sexual purposes. 
One is strictly prohibited from engaging in any activity that will potentially violate these guidelines.
Any potential commercial use of this code should be approved by the authors.