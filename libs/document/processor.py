import re
import torch

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader,TextLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import RecursiveCharacterTextSplitter  
from langchain_core.documents import Document

from libs.utils.common import get_default_rich_progress, contains_only_punctuation


class DocumentProcessor:
    # def __init__(self, embedding_model='BAAI/bge-small-zh-v1.5', device="cuda", batch_size=16, progress=None):
    def __init__(self, embedding_model='Qwen/Qwen3-Embedding-0.6B', device="cuda", batch_size=2, progress=None, logger=None, debug=False):
        self.embed_model = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': device},
            encode_kwargs={'batch_size': batch_size}
        )

        if progress is None:
            progress = get_default_rich_progress()
            progress.start()
        self.progress = progress

        self.logger = logger
        self.debug = debug

    def del_model(self):
        del self.embed_model
        torch.cuda.empty_cache()
        self.embed_model = None
        
    def detect_content_type(self, text):
        if re.search(r'["“`](.+?)[`"”]', text):
            return 'dialogue'
        return 'normal'  

    # https://zhuanlan.zhihu.com/p/27321459929
    def process_documents(self, input_dir, chunk_size=512, chunk_overlap=128):
        # Process PDF and TXT files in the input directory
        loaders = [
            # DirectoryLoader(input_dir, glob="**/*.pdf", loader_cls=PyPDFLoader),
            DirectoryLoader(input_dir, glob='**/*.txt', loader_cls=TextLoader)
        ]

        documents = []

        for loader in loaders:
            documents.extend(loader.load())


        self.logger.info(f'Loaded {len(documents)} documents.')

        chunker = SemanticChunker(
            embeddings=self.embed_model, 
            breakpoint_threshold_amount=82,  
            add_start_index=True   
        )
        base_chunks = chunker.split_documents(documents)  

        final_chunks = []

        for i, chunk in enumerate(base_chunks):
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap)

            final_chunks.extend(splitter.split_documents([chunk]))  
        
        cleaned_chunks = []

        for i, chunk in enumerate(final_chunks):
            if not contains_only_punctuation(chunk.page_content):
                cleaned_chunks.append(chunk)

        final_chunks = cleaned_chunks

        for i, chunk in enumerate(final_chunks):
            chunk.metadata.update({
                'chunk_id': f'chunk_{i}',
                'content_type': self.detect_content_type(chunk.page_content)
            })   
            
        return final_chunks
    
    def extract_contents(self, chunks, content_type):
        return [chunk for chunk in chunks if chunk['type'] == content_type]
    
    def split_utterance_into_sentences(self, utterance):
        sentences = []

        split_sentences = re.split(r'(?<=[。？！.?!])\s*', utterance)
        for s in split_sentences:
            if not contains_only_punctuation(s) and s.strip() != '':
                sentences.append(s.strip())

        return sentences
    
    def split_sentence_into_sentences(self, utterance):
        sentences = []

        split_sentences = re.split(r'(?<=[,，。？！.?!])\s*', utterance)
        for s in split_sentences:
            if not contains_only_punctuation(s) and s.strip() != '':
                sentences.append(s.strip())

        return sentences
    
    def split_utterance_into_phrases(self, utterance):
        phrases = []

        split_phrases = re.split(r'(?<=[。？！.?!,，;；])\s*', utterance)
        for s in split_phrases:
            if not contains_only_punctuation(s) and s.strip() != '':
                phrases.append(s.strip())

        return phrases
    
    def split_chunk_into_utterances(self, chunks):
        sentences = []

        for chunk in chunks:
            utterance = re.findall(r'["“](.+?)["”]', chunk.page_content)
            for sentence in utterance:
                sentences.extend(self.split_utterance_into_sentences(sentence))

        return sentences
    
    def sentences_to_chunks(self, sentences):
        chunks = []

        for i, sentence in enumerate(sentences):
            chunks.append(
                Document(
                page_content=sentence,
                metadata={'chunk_id': f"chunk_{i}"}
            ))

        return chunks
    
    def dialogues_to_sentences(self, dialogues, keep_utterance=False):
        results = {}

        task = self.progress.add_task(description=f'Extracting sentences from {len(dialogues)} dialogues', total=len(dialogues))

        for k, v in dialogues.items():
            if keep_utterance:
                sentences = v
            else:
                sentences = []
                for i, s in enumerate(v):
                    split_sentences = re.split(r'(?<=[。？！.?!])\s*', s)
                    for s in split_sentences:
                        if not contains_only_punctuation(s.strip()) and s.strip() != '':
                            sentences.append(s.strip())

            if len(sentences) > 0:
                results[k] = sentences

            self.progress.advance(task)
        
        if not self.debug:
            self.progress.update(task, visible=False)
            self.progress.refresh()

        return results
    


    
    
