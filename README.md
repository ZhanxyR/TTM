<p align="center">

  <h1 align="left">Test-Time-Matching: Decouple Personality, Memory, and Linguistic Style in LLM-based Role-Playing Language Agent</h1>
  <p align="left">
    <a href="https://zhanxy.xyz/" rel="external nofollow noopener" target="_blank"><strong>Xiaoyu Zhan</strong></a>
    ·
    <strong>Xinyu Fu</strong></a>
    ·
    <strong>Hao Sun</strong></a>
    ·
    <a href="http://www.njumeta.com/liyq/" rel="external nofollow noopener" target="_blank"><strong>Yuanqi Li</strong></a>
    ·
    <a href="https://scholar.google.com.hk/citations?user=Sx4PQpQAAAAJ&hl=en" rel="external nofollow noopener" target="_blank"><strong>Jie Guo</strong></a>
    ·
    <a href="https://cs.nju.edu.cn/ywguo/index.htm" rel="external nofollow noopener" target="_blank"><strong>Yanwen Guo*</strong></a>

  </p>
  <p align="left">
    <a href="https://arxiv.org/abs/2507.16799" rel="external nofollow noopener" target='_blank'>
        <img src="https://img.shields.io/badge/arXiv-2507.16799-B31B1B" alt='Arxiv Link'></a>
  </p>
  <br>

</p>

## Overview
- [1 - Installation](#installation)
- [2 - Demo](#demo)
- [3 - Complete Process](#process)
- [4 - Customization](#customization)
- [5 - Acknowledgments](#acknowledgments)
- [6 - Citation](#citation)
- [7 - Contact](#contact)

<a id="installation"></a>
## 📍 Installation

### 1. Get Started
Start from creating a conda environment.
```bash
git clone https://github.com/ZhanxyR/TTM.git
cd TTM
conda create -n ttm python=3.10
conda activate ttm
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

<a id="demo"></a>
## 🚀 Demo

### 1. Download Preprocessed Character Profiles and Database.
Download from [Hugging Face](https://github.com/ZhanxyR/TTM).

The completed structure should be like:

```
|-- TTM
    |-- cache
        |-- demo_Harry_Potter_1_4_Qwen25_32b_512_long
            |-- rkg_graph
                |-- relations_vdb
                |-- entities_vdb
            |-- Role Name
                |-- background.json
                |-- personality.json
                |-- linguistic_style.json   
            |-- rolse.json
            |-- roles_sentences.json
            |-- chunks.json
```

### 2. Start a Local VLLM Server (Optional).
```bash
# Modify first: supplement or adjust the necessary parameters.
sh scripts/vllm_server.sh
```

### 3. Start Chatting.
```bash
# Modify first: supplement or adjust the necessary parameters.
sh scripts/demo.sh
```

<a id="process"></a>
## 🎯 Complete Process

> [!IMPORTANT]
> The entire process involves a significant number of API calls. Please confirm that you truly intend to proceed.

### 1. Try with Our Examples.
```bash
# Modify first: supplement or adjust the necessary parameters.
sh scripts/complete_en.sh
# Or
sh scripts/complete_zh.sh
```
By default, the processed results will be saved to `cache/name` directory.

The log files consist of TTM's log (`logs/name_time.log`) and [DIGIMON](https://github.com/JayLZhou/GraphRAG)'s log (`logs/GraphRAG_Logs/time.log`).

> [!NOTE]
> For more details of our parameters, please check the [Descriptions](./docs/Parameters.md).


### 2. Run with Your Data.
You should first organize the text files as below. Then modify the input and output pathes in the script.

```
-i examples/yours
-n as_you_like

|-- TTM
    |-- examples
        |-- yours
            |-- 001.txt
            |-- 002.txt
```

<a id="customization"></a>
## 🎨 Customization

Later.

## Acknowledgments

This work was supported by the National Natural Science Foundation of China (62032011) and the Natural Science Foundation of Jiangsu Province (BK20211147).

There are also many powerful resources that greatly benefit our work:

- [LightRAG](https://github.com/HKUDS/LightRAG)
- [DIGIMON](https://github.com/JayLZhou/GraphRAG)
- [ChatHaruhi](https://github.com/LC1332/Chat-Haruhi-Suzumiya)
- [CoSER](https://github.com/Neph0s/COSER)

## Citation

```bibtex
@misc{zhan2025ttm,
      title={Test-Time-Matching: Decouple Personality, Memory, and Linguistic Style in LLM-based Role-Playing Language Agent}, 
      author={Zhan, Xiaoyu and Fu, Xinyu and Sun, Hao and Li, Yuanqi and Guo, Jie and Guo, Yanwen},
      year={2025},
      eprint={2507.16799},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2507.16799}, 
}
```


## Contact
Zhan, Xiaoyu (zhanxy@smail.nju.edu.cn) and Fu, Xinyu (xinyu.fu@smail.nju.edu.cn).

