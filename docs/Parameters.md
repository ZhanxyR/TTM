Click to see the details.

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
> [!NOTE]
> During the preprocessing stage, **the database is created only when this parameter is used**.

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
> [!NOTE]
> Chatting mode will skip all of the preprocessing function, and mandatorily load the cached files. Ensure you have prepared the necessary files.
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
> [!NOTE]
> Increasing the chunk size can accelerate the processing time.
#### --chunk_overlap
  The overlap size to be used for processing document. (`64` by default).
#### --keep_utterance
  Do not split the utterances into sentences. (`False` by default).
> [!NOTE]
> This setting controls whether to store individual sentences or complete conversation utterances. **Setting it to True is recommended** if the number of historical utterances is enough for retrieving.
#### --use_haruhi
  Whether to use [Haruhi](https://huggingface.co/silk-road/Haruhi-Dialogue-Speaker-Extract_qwen18) for dialogues extraction. (`False` by default).
#### --skip_summarize
  Skip the summarization step. (`False` by default).
#### --process_only
  Only process the documents and save the intermediate results. (`False` by default).
> [!NOTE]
> It will exit immediately after preprocessing is complete.
#### --rebuild_graphrag
  Force rebuilding the vector database. (`False` by default).
> [!IMPORTANT]
> Use with caution, as it will overwrite the cached files.
#### --ignore_cache
  Force recalculation: recalculate everything and rewrite cached data. (`False` by default).
> [!IMPORTANT]
> Use with caution, as it will overwrite the cached files.

</details>

<details>
<summary><span style="font-weight: bold;">TTM args (used in chatting)</span></summary>

#### --retriever_k_l
  The number of similar sentences to be retrieved for each linguistic style query, used for reranking. (`40` by default).
#### --memory_k
  The number of related chunks to be used for memory. (`10` by default).
#### --matching_type
  The matching type to be used for matching linguistic style query. (`dynamic` by default).
> [!NOTE]
> Select from ['simple', 'parallel', 'serial', 'dynamic']
#### --matching_k
  The number of historical utterance examples for each linguistic style query. (`15` by default).
> [!NOTE]
>  Hybrid retrieval will double the final numbers.
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