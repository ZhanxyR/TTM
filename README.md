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

> [!NOTE]
> For more details of our parameters, please click to see the details.
<details>
<summary><span style="font-weight: bold;">Specific args to be modified</span></summary>

#### --input / -i
  The directory of input documents, used for doucment processing. If document processing is no longer required, there is no need to specify it.
#### --roles / -r
  The selected roles for role-playing in the roles list, separated by comma. The role name should be defined in the roles.json file.
#### --name / -n
  The name for this experiment, used for saving and loading.
#### --cache / -c
  The cache directory to be used for saving and loading the intermediate results. (`cache` by default).
#### --url / -u
  The IP address of the LLM server.
#### --model / -m
  The model name of the LLM server.
#### --key / -k
  The API key of the LLM server.
#### --language / -l
  The language of both the input documents and the used prompts. (`zh` by default).
#### --workers / -w
  The number of workers to be used for multi-threading. (`20` by default).
#### --graphrag / -g
  Whether to use RAG for detailed memory. (`False` by default).
<div style="border-left: 4px solid #42b983; background-color: #f0fff0; padding: 10px; margin: 10px 0;">
<p><strong>Note:</strong> During the preprocessing stage, **the database is created only when this parameter is used**.</p>
</div>

#### --log
  The path to save the logs. (`logs` by default).
#### --max_tokens
  The maximum number of tokens to be used. (`2048` by default).
#### --top_p
  The top-p probability to be used. (`0.9` by default).
#### --temperature
  The temperature to be used. (`0.7` by default).

</details>

<details>
<summary><span style="font-weight: bold;">Mode args</span></summary>

## Multiple modes can be active at the same time, as they do not conflict with one another.

#### --serial
  Run in serial mode, without multi-threading. (`False` by default).
#### --debug
  Run in debug mode, with additional log infomation. (`False` by default).
#### --chat
  Run in chatting mode, do not execute any document processing. (`False` by default)
<div style="border-left: 4px solid #42b983; background-color: #f0fff0; padding: 10px; margin: 10px 0;">
<p><strong>Note:</strong> Chatting mode will skip all of the preprocessing function, and mandatorily load the cached files. Ensure you have prepared the necessary files.</p>
</div>
#### --test
  Run in test mode, with predefined user inputs rather than interaction. (`False` by default).
#### --short
  Run in short mode, the agent will generate shorter responses. (`False` by default).

</details>

<details>
<summary><span style="font-weight: bold;">Model args</span></summary>

#### --haruhi_model
  The path to the Haruhi model. Won't be used if args.use_haruhi is False. (`silk-road/Haruhi-Dialogue-Speaker-Extract_qwen18` by default).
#### --embedding_model
  The path to the embedding model. Used for utterance retrieval. (`Qwen/Qwen3-Embedding-0.6B` by default).
#### --rerank_model
  The path to the rerank model. Used for utterance retrieval. (`Qwen/Qwen3-Reranker-0.6B` by default).
#### --graph_embedding_model
  The path to the graph embedding model. Used in RAG. (`BAAI/bge-large-zh-v1.5` by default).

</details>

<details>
<summary><span style="font-weight: bold;">Preprocessing args (used in preprocessing)</span></summary>

#### --chunk_size
  The chunk size to be used for processing document. (`512` by default).
<div style="border-left: 4px solid #42b983; background-color: #f0fff0; padding: 10px; margin: 10px 0;">
<p><strong>Note:</strong> Increasing the chunk size can accelerate the processing time.</p>
</div>
#### --chunk_overlap
  The overlap size to be used for processing document. (`64` by default).
#### --keep_utterance
  Do not split the utterances into sentences. (`False` by default).
<div style="border-left: 4px solid #42b983; background-color: #f0fff0; padding: 10px; margin: 10px 0;">
<p><strong>Note:</strong> This setting controls whether to store individual sentences or complete conversation utterances. **Setting it to True is recommended** if the number of historical utterances is enough for retrieving.</p>
</div>
#### --use_haruhi
  Whether to use [Haruhi](https://huggingface.co/silk-road/Haruhi-Dialogue-Speaker-Extract_qwen18) for dialogues extraction. (`False` by default).
#### --skip_summarize
  Skip the summarization step. (`False` by default).
#### --process_only
  Only process the documents and save the intermediate results. (`False` by default).
<div style="border-left: 4px solid #42b983; background-color: #f0fff0; padding: 10px; margin: 10px 0;">
<p><strong>Note:</strong> It will exit immediately after preprocessing is complete.</p>
</div>
#### --rebuild_graphrag
  Force rebuilding the vector database. (`False` by default).
<div style="border-left: 4px solid #ff6666; background-color: #fff0f0; padding: 10px; margin: 10px 0;">
<p><strong>Important:</strong> Use with caution, as it will overwrite the cached files.</p>
</div>
#### --ignore_cache
  Force recalculation: recalculate everything and rewrite cached data. (`False` by default).
<div style="border-left: 4px solid #ff6666; background-color: #fff0f0; padding: 10px; margin: 10px 0;">
<p><strong>Important:</strong> Use with caution, as it will overwrite the cached files.</p>
</div>

</details>

<details>
<summary><span style="font-weight: bold;">TTM args (used in chatting)</span></summary>

#### --retriever_k_l
  The number of similar sentences to be retrieved for each linguistic style query, used for reranking. (`40` by default).
#### --memory_k
  The number of related chunks to be used for memory. (`10` by default).
#### --matching_type
  The matching type to be used for matching linguistic style query. (`dynamic` by default).
<div style="border-left: 4px solid #42b983; background-color: #f0fff0; padding: 10px; margin: 10px 0;">
<p><strong>Note:</strong> Select from ['simple', 'parallel', 'serial', 'dynamic'].</p>
</div>
#### --matching_k
  The number of historical utterance examples for each linguistic style query. (`15` by default).
<div style="border-left: 4px solid #42b983; background-color: #f0fff0; padding: 10px; margin: 10px 0;">
<p><strong>Note:</strong> Hybrid retrieval will double the final numbers.</p>
</div>
#### --max_common_words
  The maximum number of common words of each type to be used for matching the linguistic style query. (`20` by default).
#### --use_clean
  Remove the linguistic style of the utterance when matching. (`False` by default).
#### --clean_first_only
  Only remove the linguistic style of the first time response (not the styleless response) during chatting. (`False` by default).
#### --split_sentence
  Split the sentence into sentences by comma for matching. (`False` by default).
#### --disable_action
  Disable the action display during chatting. (`False` by default).
#### --disable_personality
  Disable the personality setting during chatting.
#### --disable_background
  Disable the background setting during chatting.
#### --disable_linguistic_preference
  Disable the linguistic preference setting during chatting.
#### --disable_common_words
  Disable the common words setting during chatting.
#### --disable_matching
  Disable the linguistic style matching during chatting.

</details>

<a id="process"></a>
## 🎯 Complete Process

> [!IMPORTANT]
> The entire process involves **a significant number of API calls**. Please confirm that you truly intend to proceed.

### 1. Try with Our Examples.
```bash
# Modify first: supplement or adjust the necessary parameters.
sh scripts/complete_en.sh
# Or
sh scripts/complete_zh.sh
```
By default, the processed results will be saved to `cache/name` directory.

The log files consist of TTM's log (`logs/name_time.log`) and [DIGIMON](https://github.com/JayLZhou/GraphRAG)'s log (`logs/GraphRAG_Logs/time.log`).


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
If you find this work helpful, please consider citing our paper.

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

