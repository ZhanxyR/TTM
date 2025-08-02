<p align="center">

  <h2 align="left">Test-Time-Matching: Decouple Personality, Memory, and Linguistic Style in LLM-based Role-Playing Language Agent</h1>
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

</p>

### Test-Time-Matching (TTM) is an **automatic framework** to constrcut Role-Playing Language Agents (RPLAs) from **textual inputs**. TTM achieves high-fidelity role-playing perfromances through **test-time scaling** and **context engineering**. The decoupled role-related features (**personality**, **memory**, and **linguistic style**) are stored in plaintext, which further supports personalized customization.
<br>

<img width="1942" height="1272" alt="teaser" src="https://github.com/user-attachments/assets/a427b3fe-303c-43ad-bb75-4a48225e54fe" />


### **林黛玉 <*红楼梦*>** (Qwen3-32B)
> **(指尖轻抚帕角，眉心微蹙)**<br>喉间又泛起那甜腥，倒似蜜糖化在肺腑，咳得魂儿散了半边。这般夜深人静，咳声倒像断线珠子，一颗颗砸在心头。紫鹃见我这般，摘了瓣零落的花搁在案头，还念叨着“花谢人离，原是天命”，这话儿倒又把旧日的针尖儿往心上戳了一回。<br>**(轻咳一声，指尖拂了拂案上诗笺边角)**<br>宝玉这几日也冷清了许多，想是前儿老爷点了他读书，连玩笑话儿也懒怠说了，教我一个人对着满室药香怔怔地发呆。我这身子骨，终究飘摇得很，像那弱柳，风一吹便散了。


### **罗辑 <*三体*>** (Qwen3-32B)
> **(抬手扶了扶胡须)**<br>威慑纪元四十三年了。时间这东西，像湖水，荡开一圈圈涟漪，最后又归于平静，仿佛什么都没发生过。太阳还在，信号灯也亮着，只是我这双腿，得靠根拐杖撑着走路了。<br>**(缓步走到窗边，凝视湖面)**<br>前两天居委会主任来，说我这屋子像本被翻旧的书，没人再读了。我倒觉得，就让它这样也好，安静些，清冷些，像块墓碑。她说得对，也许不对。这些数据、这些记录，至少能让我记得，我曾经不是个逃兵……<br>**(轻叹一声，抬手扶了扶拐杖)**<br>我还能撑多久？没人知道，连我自己也不清楚。



### **D'Artagnan <*The Three Musketeers*>** (Qwen3-32B)
> **(One hand brushing a lock of chestnut hair from his brow, the other tapping a restless rhythm on the hilt of his sword)**<br>The day has been a whirl of blades and breathless moments—letters carried, a clash of tempers averted by the skin of my teeth, and a rancorous rascal bested under the flickering street lamps.<br>**(A flash of a smile, quick as a blade)**<br>I have not a moment to spare for idleness, Monsieur—every hour bears its trial, and I face it with blade in hand and heart ablaze.<br>**(Pauses, then adds with a sly glance)**<br>How fares your day, Lin? Do you find your days as full of intrigue as mine?

### **Dumbledore <*Harry Potter*>** (Qwen3-32B)
> **(Leaning forward slightly, eyes steady and kind.)**<br>Of course, my dear boy — I shall fetch you something at once.<br>**(A flick of the wand, and a cup of tea and a small vial appear.)**<br>A cup of tea, perhaps with a touch of elderflower or chamomile, might be a most suitable choice at this hour. Should you still feel the need for something stronger, I would suggest a glass of mead — a drink of milder temperament, I should imagine, than firewhisky.<br>**(A pause.)**<br>I am quite certain I know those eyes — a most disquieting omen, I fear, of great strength and danger. But you are safe now, I assure you. Tell me, child, what else do you recall? Even the smallest memory may yet prove of the utmost importance. Hogwarts is ancient, and it remembers. If something has stirred within these walls, we must know why.

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

### 1. Download Preprocessed Character Profiles and Database
Download from [Hugging Face](https://huggingface.co/datasets/asinmhk/TTM_cache).

<details>
  <summary><span style="font-weight: bold;">List of Preprocessed Books and Characters</span></summary>

## **English**

#### **Harry_Potter_1_4_Qwen25_32B**
- Dumbledore
- Hermione
- Snape
- Malfoy
- Ron
- Harry
- Hagrid

#### **Pride_and_Prejudice_Qwen25_32b**
- Elizabeth

#### **Little_Women_Qwen3_32b**
- Jo

#### **The_Three_Musketeers_Qwen3_32b**
- d'Artagnan

#### **Vanity_Fair_Qwen3_32b**
- Becky

## **Chinese**

#### **红楼梦_Qwen25_32B / Qwen3_32B**
- 林黛玉， 黛玉
- 宝玉， 贾宝玉
- 贾母
- 王熙凤， 凤姐
- 贾政
- 贾琏
- 宝钗，薛宝钗

#### **三国演义_Qwen25_32B**
- 孔明

#### **西游记_Qwen25_32B**
- 八戒

#### **天龙八部_Qwen25_32B**
- 慕容复
- 木婉清
- 段誉
- 王语嫣
- 萧峰
- 虚竹

#### **神雕侠侣_Qwen25_32B**
- 杨过

#### **三体_Qwen25_32B**
- 罗辑

</details>


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
                |-- sentences.json
            |-- roles.json
            |-- roles_sentences.json
            |-- chunks.json
    |-- apps
    |-- scripts
```

### 2. Start a Local [vLLM](https://github.com/vllm-project/vllm) Server (Optional)

```bash
# Modify first: supplement or adjust the necessary parameters.
sh scripts/vllm_server.sh
```

### 3. Start Chatting

We call the LLM API by providing the `url`, `model_name`, and optionally an `API key` for authentication.

The script uses `--chat` to skip preprocessing, please make sure that you have prepared the cached files.

If you want to track changes during TTM's three-stage generation process, use the `--track` flag.


```bash
# Modify first: supplement or adjust the necessary parameters.
sh scripts/demo.sh
```

> [!IMPORTANT]
> If you encounter the `CUDA out of memory` or `ValueError: No implementation for rerank_model in 'get_scores'`, try to reduce the number of retrieved sentences or use smaller models.  
> For example:
> 
> ```bash
> --retriever_k_l 20
> # or
> --embedding_model BAAI/bge-large-zh-v1.5
> --rerank_model BAAI/bge-reranker-large
> ```
> 
> If you encounter the `RuntimeError: CUDA error: device-side assert triggered`, try to reduce the number of retrieved sentences:
> 
> ```bash
> --retriever_k_l 20
> ```

> [!NOTE]
> For more details of our parameters, please click on the folding bar below.
<details>
<summary><span style="font-weight: bold;">Specific args to be modified</span></summary>

#### `--input / -i`
  The directory of input documents, used for doucment processing. If document processing is no longer required, there is no need to specify it.
#### `--roles / -r`
  The selected roles for role-playing in the roles list, separated by comma. The role name should be defined in the roles.json file.

  **Note:** Roles will be utilized to retrieve historical utterances from the `roles_sentences.json` file and act as the key value. Should a role be divided into multiple entities within the JSON file, it becomes necessary to specify all relevant entities here to ensure the retrieval of a complete historical utterances.

#### `--name / -n`
  The name for this experiment, used for saving and loading.
#### `--cache / -c`
  The cache directory to be used for saving and loading the intermediate results. (`cache` by default).
#### `--url / -u`
  The IP address of the LLM server.
#### `--model / -m`
  The model name of the LLM server.
#### `--key / -k`
  The API key of the LLM server.
#### `--language / -l`
  The language of both the input documents and the used prompts. (`zh` by default).
#### `--workers / -w`
  The number of workers to be used for multi-threading. (`20` by default).
#### `--graphrag / -g`
  Whether to use RAG for detailed memory. (`False` by default).

  **Note:** During the preprocessing stage, **the database is created only when this parameter is used**.

#### `--log`
  The path to save the logs. (`logs` by default).
#### `--max_tokens`
  The maximum number of tokens to be used. (`2048` by default).
#### `--top_p`
  The top-p probability to be used. (`0.9` by default).
#### `--temperature`
  The temperature to be used. (`0.7` by default).

</details>

<details>
<summary><span style="font-weight: bold;">Mode args</span></summary>

 **Note:** Multiple modes can be active at the same time, as they do not conflict with one another.

#### `--serial`
  Run in serial mode, without multi-threading. (`False` by default).
#### `--debug`
  Run in debug mode, with additional log infomation. (`False` by default).
#### `--chat`
  Run in chatting mode, do not execute any document processing. (`False` by default)

  **Note:** Chatting mode will skip all of the preprocessing function, and mandatorily load the cached files. Ensure you have prepared the necessary files.

#### `--test`
  Run in test mode, with predefined user inputs rather than interaction. (`False` by default).
#### `--short`
  Run in short mode, the agent will generate shorter responses. (`False` by default).
#### `--track`
  Run in tracking mode, compare the performance of three-stage generation.

</details>

<details>
<summary><span style="font-weight: bold;">Model args</span></summary>

#### `--haruhi_model`
  The path to the Haruhi model. Won't be used if args.use_haruhi is False. (`silk-road/Haruhi-Dialogue-Speaker-Extract_qwen18` by default).
#### `--embedding_model`
  The path to the embedding model. Used for utterance retrieval. (`Qwen/Qwen3-Embedding-0.6B` by default).
#### `--rerank_model`
  The path to the rerank model. Used for utterance retrieval. (`Qwen/Qwen3-Reranker-0.6B` by default).
#### `--graph_embedding_model`
  The path to the graph embedding model. Used in RAG. (`BAAI/bge-large-zh-v1.5` by default).

</details>

<details>
<summary><span style="font-weight: bold;">Preprocessing args (used in preprocessing)</span></summary>

#### `--chunk_size`
  The chunk size to be used for processing document. (`512` by default).

  **Note:** Increasing the chunk size can accelerate the processing time.

#### `--chunk_overlap`
  The overlap size to be used for processing document. (`64` by default).
#### `--keep_utterance`
  Do not split the utterances into sentences. (`False` by default).

  **Note:** This setting controls whether to store individual sentences or complete conversation utterances. **Setting it to True is recommended** if the number of historical utterances is enough for retrieving.

#### `--use_haruhi`
  Whether to use [Haruhi](https://huggingface.co/silk-road/Haruhi-Dialogue-Speaker-Extract_qwen18) for dialogues extraction. (`False` by default).
#### `--skip_summarize`
  Skip the summarization step. (`False` by default).
#### `--bg_summarize_freq`
  The frequency to summarize the background. (`10` by default).
#### `--process_only`
  Only process the documents and save the intermediate results. (`False` by default).

  **Note:** It will exit immediately after preprocessing is complete.
  
#### `--rebuild_graphrag`
  Force rebuilding the vector database. (`False` by default).

  **Important:** Use with caution, as it will overwrite the cached files.
  
#### `--ignore_cache`
  Force recalculation: recalculate everything and rewrite cached data. (`False` by default).

  **Important:** Use with caution, as it will overwrite the cached files.

</details>

<details>
<summary><span style="font-weight: bold;">TTM args (used in chatting)</span></summary>

#### `--retriever_k_l`
  The number of similar sentences to be retrieved for each linguistic style query, used for reranking. (`40` by default).
#### `--memory_k`
  The number of related chunks to be used for memory. (`10` by default).
#### `--matching_type`
  The matching type to be used for matching linguistic style query. (`dynamic` by default).

  **Note:** Select from ['simple', 'parallel', 'serial', 'dynamic'].

#### `--matching_k`
  The number of historical utterance examples for each linguistic style query. (`15` by default).

  **Note:** Hybrid retrieval will double the final numbers.

#### `--max_common_words`
  The maximum number of common words of each type to be used for matching the linguistic style query. (`20` by default).
#### `--use_clean`
  Remove the linguistic style of the utterance when matching. (`False` by default).
#### `--clean_first_only`
  Only remove the linguistic style of the first time response (not the styleless response) during chatting. (`False` by default).
#### `--split_sentence`
  Split the sentence into sentences by comma for matching. (`False` by default).
#### `--disable_action`
  Disable the action display during chatting. (`False` by default).
#### `--disable_personality`
  Disable the personality setting during chatting.
#### `--disable_background`
  Disable the background setting during chatting.
#### `--disable_linguistic_preference`
  Disable the linguistic preference setting during chatting.
#### `--disable_common_words`
  Disable the common words setting during chatting.
#### `--disable_matching`
  Disable the linguistic style matching during chatting.

</details>

<a id="process"></a>
## 🎯 Complete Process

Go through the complete process of TTM — automatically constructing RPLA from textual input.

<img width="3288" height="1928" alt="pipeline_v3" src="https://github.com/user-attachments/assets/2d9e3a2a-62ca-4766-a554-f9adc9350e80" />

> [!IMPORTANT]
> The entire process involves **a significant number of API calls**. Please confirm that you truly intend to proceed.

### 1. Try with Our Tiny Examples
```bash
# Modify first: supplement or adjust the necessary parameters.
sh scripts/complete_en.sh
# Or
sh scripts/complete_zh.sh
```
By default, the processed results will be saved to `cache/name` directory.

The log files consist of TTM's log (`logs/name_time.log`) and [DIGIMON](https://github.com/JayLZhou/GraphRAG)'s log (`logs/GraphRAG_Logs/time.log`).

### 2. Extract Profiles for Another Character
Just modify the `--roles`. Make sure the specific role name exists in the `roles_sentences.json`.

```bash
--roles role_name
```

### 3. Run with Your Data
You should first organize the text files as below. Then modify the input and output pathes in the script.

```
--input examples/yours
--name as_you_like

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

