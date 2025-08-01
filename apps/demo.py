import argparse
import os
import time
import re

# from huggingface_hub import login

from libs.document.processor import DocumentProcessor
from libs.retriever.simple_retriever import Retriever
from libs.llm.base import LLM
from libs.utils.logger import get_logger, remove_handlers
from libs.utils.common import *


os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  
# hf_token = "your_hf_token"
# login(hf_token)
 
def parse():
    parser = argparse.ArgumentParser()

    # Specific args to be modified
    parser.add_argument('-i', '--input', type=str, default='examples/yours', help='The directory of input documents, used for doucment processing. If document processing is no longer required, there is no need to specify it.')
    parser.add_argument('-r', '--roles', type=str, default='roles', help='The selected roles for role-playing in the roles list, separated by comma.  The role name should be defined in the roles.json file.')
    parser.add_argument('-n', '--name', type=str, default='demo_test', help='The name for this experiment, used for saving and loading.')
    parser.add_argument('-c', '--cache', type=str, default='cache', help='The cache directory to be used for saving and loading the intermediate results.')
    parser.add_argument('-u', '--url', type=str, default='http://0.0.0.0:8000/v1', help='The IP address of the LLM server.')
    parser.add_argument('-m', '--model', type=str, default='Qwen3-32B', help='The model name of the LLM server.')
    parser.add_argument('-k', '--key', type=str, default='EMPTY', help='The API key of the LLM server.')
    parser.add_argument('-l', '--language', type=str, default='zh', help='The language of both the input documents and the used prompts.', choices=['zh', 'en'])
    parser.add_argument('-w', '--workers', type=int, default=20, help='The number of workers to be used for multi-threading.')
    parser.add_argument('-g', '--graphrag', action='store_true', default=False, help='Whether to use RAG for detailed memory. During the preprocessing stage, the database is created only when this parameter is used')

    parser.add_argument('--log', type=str, default='logs', help='The path to save the logs.')
    parser.add_argument('--max_tokens', type=int, default=2048, help='The maximum number of tokens to be used.')
    parser.add_argument('--top_p', type=float, default=0.9, help='The top-p probability to be used.')
    parser.add_argument('--temperature', type=float, default=0.7, help='The temperature to be used.')

    # Mode args
    # Multiple modes can be active at the same time, as they do not conflict with one another.
    parser.add_argument('--serial', action='store_true', default=False, help='Run in serial mode, without multi-threading.')
    parser.add_argument('--debug', action='store_true', default=False, help='Run in debug mode, with additional log infomation.')
    parser.add_argument('--chat', action='store_true', default=False, help='Run in chatting mode, do not execute any document processing.')
    parser.add_argument('--test', action='store_true', default=False, help='Run in test mode, with predefined user inputs rather than interaction.')
    parser.add_argument('--short', action='store_true', default=False, help='Run in short mode, the agent will generate shorter responses.')
    parser.add_argument('--track', action='store_true', default=False, help='Run in tracking mode, compare the performance of three-stage generation.')

    # Model args
    parser.add_argument('--haruhi_model', type=str, default='silk-road/Haruhi-Dialogue-Speaker-Extract_qwen18', help='The path to the Haruhi model. Won\'t be used if args.use_haruhi is False.')
    parser.add_argument('--embedding_model', type=str, default='Qwen/Qwen3-Embedding-0.6B', help='The path to the embedding model.  Used for utterance retrieval.')
    parser.add_argument('--rerank_model', type=str, default='Qwen/Qwen3-Reranker-0.6B', help='The path to the rerank model.  Used for utterance retrieval.')
    parser.add_argument('--graph_embedding_model', type=str, default='BAAI/bge-large-zh-v1.5', help='The path to the graph embedding model. Used in RAG.')

    # Preprocessing args
    parser.add_argument('--chunk_size', type=int, default=512, help='The chunk size to be used for processing document.')
    parser.add_argument('--chunk_overlap', type=int, default=64, help='The overlap size to be used for processing document.')
    parser.add_argument('--keep_utterance', action='store_true', default=False, help='Do not split the utterances into sentences. This setting controls whether to store individual sentences or complete conversation utterances. Setting it to True is recommended if the number of historical utterances is enough for retrieving.')
    parser.add_argument('--use_haruhi', action='store_true', default=False, help='Whether to use Haruhi for dialogues extraction.')
    parser.add_argument('--skip_summarize', action='store_true', default=False, help='Skip the summarization step.')
    parser.add_argument('--bg_summarize_freq', type=int, default=10, help='The frequency to summarize the background.')
    parser.add_argument('--process_only', action='store_true', default=False, help='Only process the documents and save the intermediate results.')
    parser.add_argument('--rebuild_graphrag', action='store_true', default=False, help='Force rebuilding the vector database. Use with caution, as it will overwrite the cached files.')
    parser.add_argument('--ignore_cache', action='store_true', default=False, help='Force recalculation: recalculate everything and rewrite cached data. Use with caution, as it will overwrite the cached files.')

    # TTM args
    parser.add_argument('--retriever_k_l', type=int, default=40, help='The number of similar sentences to be retrieved for each linguistic style query, used for reranking.')
    parser.add_argument('--memory_k', type=int, default=10, help='The number of related chunks to be used for memory.')
    parser.add_argument('--matching_type', type=str, default='dynamic', help='The matching type to be used for matching linguistic style query.', choices=['simple', 'parallel', 'serial', 'dynamic'])
    parser.add_argument('--matching_k', type=int, default=15, help='The number of historical utterance examples for each linguistic style query.')
    parser.add_argument('--max_common_words', type=int, default=20, help='The maximum number of common words of each type to be used for matching the linguistic style query.')
    parser.add_argument('--use_clean', action='store_true', default=False, help='Remove the linguistic style of the utterance when matching.')
    parser.add_argument('--clean_first_only', action='store_true', default=False, help='Only remove the linguistic style of the first time response (not the styleless response) during chatting.')
    parser.add_argument('--split_sentence', action='store_true', default=False, help='Split the sentence into sentences by comma for matching.')

    parser.add_argument('--disable_action', action='store_true', default=False, help='Disable the action display during chatting.')
    parser.add_argument('--disable_personality', action='store_true', default=False, help='Disable the personality setting during chatting.')
    parser.add_argument('--disable_background', action='store_true', default=False, help='Disable the background setting during chatting.')
    parser.add_argument('--disable_linguistic_preference', action='store_true', default=False, help='Disable the linguistic preference setting during chatting.')
    parser.add_argument('--disable_common_words', action='store_true', default=False, help='Disable the common words setting during chatting.')
    parser.add_argument('--disable_matching', action='store_true', default=False, help='Disable the linguistic style matching during chatting.')

    return parser.parse_args()

def start_progress():
    progress = get_default_rich_progress()
    progress.start()
    return progress

def recalculate(ignore_cache, pathes):
    if ignore_cache:
        return True
    
    for path in pathes:
        if not os.path.exists(path):
            return True
    
    return False

def process_documents(args, root_dir, logger, processor, model, haruhi_processor=None):
    chunks_path = os.path.join(root_dir, 'chunks.json')
    sentences_path = os.path.join(root_dir,'sentences.json')

    # chunks
    if (not args.chat) and recalculate(args.ignore_cache, [chunks_path]):
        logger.info(f'Proccess documents and extract chunks from \'{args.input}\'.')
        chunks = processor.process_documents(args.input, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
        chunks = model.summarize_from_chunks(chunks, skip=args.skip_summarize)
        chunks = document_to_flatten_json(chunks)  

        save_to_flatten_json(chunks, chunks_path)
        logger.info(f'Summarize \'{len(chunks)}\' chunks. Save to \'{chunks_path}\'')
    else:
        chunks = read_from_flatten_json(chunks_path)
        logger.info(f'Load cached \'{len(chunks)} chunks\' from \'{chunks_path}\'.')

    # sentences
    if (not args.chat) and recalculate(args.ignore_cache, [sentences_path]):
        dialogue_chunks = processor.extract_contents(chunks, 'dialogue')

        if not args.use_haruhi: 
            dialogues = model.extract_dialogues(dialogue_chunks)
            sentences = processor.dialogues_to_sentences(dialogues, args.keep_utterance)
        else:
            logger.info(f'Use \'Haruhi\' for dialogue extraction.')
            dialogues = haruhi_processor.extract_dialogues(chunks=dialogue_chunks)
            dialogues = haruhi_processor.haruhi_to_dict(dialogues)
            sentences = processor.dialogues_to_sentences(dialogues, args.keep_utterance)

        save_to_json(sentences, sentences_path)
        logger.info(f'Extract \'{len(sentences)}\' entities from \'{len(dialogue_chunks)}\' dialogue chunks. Save to \'{sentences_path}\'.')
    else:
        sentences = read_from_json(sentences_path)
        logger.info(f'Load cached \'{len(sentences)} entities\' from \'{sentences_path}\'.')
    
    return chunks, sentences

def process_sentences(args, root_dir, logger, model, sentences):
    entities_path = os.path.join(root_dir, 'entities.json')
    non_entities_path = os.path.join(root_dir, 'non_entities.json')
    roles_path = os.path.join(root_dir, 'roles.json')
    entities_mapping_path = os.path.join(root_dir, 'entities_mapping.json')
    roles_indexes_path = os.path.join(root_dir, 'roles_indexes.json')
    roles_sentences_num_path = os.path.join(root_dir, 'roles_sentences_num.json')
    roles_sentences_path = os.path.join(root_dir, 'roles_sentences.json')

    # entities
    if (not args.chat) and recalculate(args.ignore_cache, [entities_path]):
        entities, non_entities = model.detect_role_entities(sentences.keys())

        save_to_json(entities, entities_path)
        save_to_json(non_entities, non_entities_path)
        logger.info(f'Extract \'{len(entities)} entities\' and \'{len(non_entities)} non-entities\' from \'{len(sentences)} sentences\'. Save to \'{entities_path}.\'')
    else:
        entities = read_from_json(entities_path)
        logger.info(f'Load cached \'{len(entities)} entities\' from \'{entities_path}\'.')

    # roles
    if (not args.chat) and recalculate(args.ignore_cache, [roles_path, entities_mapping_path, roles_indexes_path]):
        roles, entities_mapping, indexes = model.combine_roles_from_entities(entities)

        save_to_json(roles, roles_path)
        save_to_json(entities_mapping, entities_mapping_path)
        save_to_json(indexes, roles_indexes_path)

        logger.info(f'Extract \'{len(roles)} roles\' from \'{len(entities)} entities\'. Save to \'{roles_path}\'.')
    
    # roles_sentences
    if (not args.chat) and recalculate(args.ignore_cache, [roles_sentences_num_path, roles_sentences_path]):
        roles = read_from_json(roles_path)
        entities_mapping = read_from_json(entities_mapping_path)
        indexes = read_from_json(roles_indexes_path)

        roles_sentences_num = {}
        roles_sentences = {}
        for role in roles:
            role_indexes = indexes[role]
            
            for i in role_indexes:
                for s in sentences[i]:
                    roles_sentences.setdefault(role, set()).add(s)

            roles_sentences[role] = list(roles_sentences[role])
            roles_sentences_num[role] = len(roles_sentences[role])

        save_to_json(roles_sentences_num, roles_sentences_num_path)
        save_to_json(roles_sentences, roles_sentences_path)
        logger.info(f'Extract \'roles_sentences\' for \'{len(roles)} roles\'. Save to \'{roles_sentences_path}\'.')
    else:
        roles_sentences = read_from_json(os.path.join(root_dir, 'roles_sentences.json'))
        logger.info(f'Load cached \'{len(roles_sentences.items())} roles\' from \'{os.path.join(root_dir,"roles_sentences.json")}\'.')

    return roles_sentences

def process_role(args, root_dir, logger, model, chunks, selected_sentences, role_name):
    linguistic_style_path = os.path.join(root_dir, 'linguistic_style.json')
    related_chunks_indexes_path = os.path.join(root_dir,'related_chunks_indexes.json')
    personality_path = os.path.join(root_dir, 'personality.json')
    background_path = os.path.join(root_dir, 'background.json')

    # linguistic_style
    if (not args.chat) and recalculate(args.ignore_cache, [linguistic_style_path]):
        linguistic_style = model.analyze_linguistic_style_from_sentences(selected_sentences)

        save_to_json(linguistic_style, linguistic_style_path)
        logger.info(f'Analyze linguistic style from \'{len(selected_sentences)}\' sentences. Save to \'{linguistic_style_path}\'.')
    else:
        linguistic_style = read_from_json(linguistic_style_path)
        logger.info(f'Load cached \'linguistic_style\' from \'{linguistic_style_path}\'.')

    # related_chunks_indexes
    if (not args.chat) and recalculate(args.ignore_cache, [related_chunks_indexes_path]):
        related_chunks_indexes = model.get_related_chunks(chunks, role_name)

        save_to_json(related_chunks_indexes, related_chunks_indexes_path)
        logger.info(f'Get \'{len(related_chunks_indexes)}\' related chunks for role \'{role_name}\'. Save to \'{related_chunks_indexes_path}\'.')
    else:
        related_chunks_indexes = read_from_json(related_chunks_indexes_path)
        logger.info(f'Load cached \'{len(related_chunks_indexes)} related_chunks\' from \'{related_chunks_indexes_path}\'.')

    related_chunks = [chunks[i] for i in related_chunks_indexes]

    # personality
    if (not args.chat) and recalculate(args.ignore_cache, [personality_path]):
        personality, characters, related_chunks_p = model.analyze_personality_from_chunks(related_chunks, role_name)
        p_c = {'personality': personality, 'characters': characters}
        # personality_related_chunks_indexes = [related_chunks_indexes[i] for i in related_chunks_p]

        save_to_json(p_c, personality_path)
        logger.info(f'Analyze personality. Save to \'{personality_path}\'.')
    else:
        p_c = read_from_json(personality_path)
        personality = p_c['personality']
        logger.info(f'Load cached \'personality\' from \'{personality_path}\'.')

    # background
    if (not args.chat) and recalculate(args.ignore_cache, [background_path]):
        background = model.extract_background_from_chunks(related_chunks, role_name, args.bg_summarize_freq)

        save_to_json(background, background_path)
        logger.info(f'Extract background. Save to \'{background_path}\'.')
    else:
        background = read_from_json(background_path)
        logger.info(f'Load cached \'background\' from \'{background_path}\'.')

    return linguistic_style, personality, background

def process_role_sentences(args, root_dir, logger, roles_sentences, selected_roles):
    confirmed_roles = []
    selected_sentences = []

    use_cache = False

    # collect sentences for selected roles
    for role in selected_roles:
        # prioritize cache usage
        if not recalculate(args.ignore_cache, [os.path.join(root_dir, role, 'sentences.json')]):
            confirmed_roles = [role]
            use_cache = True
            break

        if role not in roles_sentences:
            logger.warning(f'Role \'{role}\' is not found in the role list.')
            continue
        
        if role not in confirmed_roles:
            confirmed_roles.append(role)
            selected_sentences += roles_sentences[role]

    role_sentences_path = os.path.join(root_dir, confirmed_roles[0], 'sentences.json')

    if (not args.chat) and recalculate(args.ignore_cache, [role_sentences_path]):
        save_to_json(selected_sentences, role_sentences_path)
        logger.info(f'Save selected sentences to \'{role_sentences_path}\'.')
    elif use_cache:
        selected_sentences = read_from_json(role_sentences_path)
        logger.info(f'Load cached selected sentences from \'{role_sentences_path}\'.')

    return selected_sentences, confirmed_roles


if __name__ == '__main__':

    args = parse()

    root_dir = os.path.join(args.cache, args.name)

    # create logger
    if not os.path.exists(args.log):
        os.makedirs(args.log)
    log_file = os.path.join(args.log, args.name + '_' + time.strftime('%Y%m%d_%H%M%S', time.localtime()) + '.log')
    if args.debug:
        log_file = log_file.replace(args.name + '_', args.name + '_debug_')
    remove_handlers()
    logger = get_logger(log_file, logger_name='TTM')
    logger.info(f'Save logs to \'{log_file}\'.')
    logger.info(f'Arguments: {args}')

    # create rich progress
    progress = start_progress()

    # initialization
    processor = DocumentProcessor(embedding_model=args.embedding_model, progress=progress, logger=logger, debug=args.debug)
    model = LLM(language=args.language, workers=args.workers, logger=logger, progress=progress, serial=args.serial, debug=args.debug)
    model.create_from_url(args.model, args.key, args.url, max_tokens=args.max_tokens, top_p=args.top_p, temperature=args.temperature)

    if args.use_haruhi:
        from libs.document.haruhi_processor import HaruhiProcessor
        haruhi_processor = HaruhiProcessor(model=args.haruhi_model, progress=progress, logger=logger, debug=args.debug)
    else:
        haruhi_processor = None

    try:

        model.init_role_generation()

        # process documents from args.input
        chunks, sentences = process_documents(args, root_dir, logger, processor, model, haruhi_processor)

        processor.del_model() # will not be used anymore, release to reduce memory usage

        # process sentences from chunks
        roles_sentences = process_sentences(args, root_dir, logger, model, sentences)

        # build graphrag from chunks
        if args.graphrag :
            model.build_graphrag(dataset_path=os.path.join(root_dir, 'chunks.json'), working_dir=root_dir, dataset_name=args.name, rebuild=(args.ignore_cache or args.rebuild_graphrag), embedding_model=args.graph_embedding_model, max_concurrent = 1 if args.serial else args.workers, chunk_size=args.chunk_size+args.chunk_overlap, chat=args.chat)
            logger.info(f'Build graphrag from chunks. Save to \'{os.path.join(root_dir, "rkg_graph")}\'.')

        selected_roles = re.split(r'[,，]', args.roles)
        selected_roles = [r.strip() for r in selected_roles]

        # get selected sentences for selected roles
        selected_sentences, confirmed_roles = process_role_sentences(args, root_dir, logger, roles_sentences, selected_roles)
    
        logger.info(f'Using \'{len(selected_sentences)}\' sentences of roles: {confirmed_roles}')

        if len(selected_sentences) == 0:
            raise ValueError(f'No sentences found for the selected roles: {selected_roles}. Please check the roles list: \'{os.path.join(root_dir, "roles.json")}\' and the sentences list: \'{os.path.join(root_dir, "roles_sentences.json")}\'.')
        
        connected_info = ', 又名' if args.language == 'zh' else ', also known as'
        role_name = f'{confirmed_roles[0]}{connected_info}{",".join(confirmed_roles[1:])}' if len(confirmed_roles) > 1 else confirmed_roles[0]

        logger.info(f'Process for role: \'{role_name}\'')

        # process role information
        linguistic_style, personality, background = process_role(args, os.path.join(root_dir, confirmed_roles[0]), logger, model, chunks, selected_sentences, role_name)
        
        progress.stop()

        if args.process_only:
            logger.info('Program terminated. The \'--process_only\' option prevents role-playing.')
            exit()

        logger.info(f'Create the TTM vector database with \'{len(selected_sentences)}\' sentences.')
        if len(selected_sentences) <= args.matching_k:
            logger.warning(f'The number of selected sentences ({len(selected_sentences)}) is less than or equal to the matching_k ({args.matching_k}). That means all the sentences will be used for matching.')

        selected_sentences = processor.sentences_to_chunks(selected_sentences)

        # build vector database for linguistic matching
        if not args.disable_matching:
            linguistic_retriever = Retriever(selected_sentences, embedding_model=args.embedding_model, rerank_model=args.rerank_model, k=args.retriever_k_l, cached_dir=os.path.join(root_dir, 'linguistic_retriever'), debug=args.debug)
        else:
            linguistic_retriever = None

        model.init_role_playing(role_name, linguistic_retriever, processor, personality=personality, background=background, linguistic_style=linguistic_style, \
                                memory_k=args.memory_k, matching_type=args.matching_type, matching_k=args.matching_k, max_common_words=args.max_common_words, \
                                short_response=args.short, track_response=args.track, use_clean=args.use_clean, clean_first_only=args.clean_first_only, split_sentence=args.split_sentence, \
                                disable_action=args.disable_action, disable_personality=args.disable_personality, disable_background=args.disable_background, \
                                disable_linguistic_preference=args.disable_linguistic_preference, disable_common_words=args.disable_common_words, disable_matching=args.disable_matching)
        
        # model.set_seed()

        logger.info(f'Role playing starts: {role_name}')

        # short response
        extra_prompt = ''
        if args.short:
            extra_prompt = ' (简短回复)' if args.language == 'zh' else ' (Short response)'

        # interactive mode
        if not args.test:
            logger.info('Enter interactive mode. Type \'exit\' to quit.')
            while True:
                user_input = input("User: ")
                if user_input.lower() in ['exit', 'quit']:
                    logger.info("Exit interactive chat.")
                    break

                processed_input = f'{user_input}{extra_prompt}'
                target = model.chat(processed_input)

                logger.info('----------- Question -----------')
                logger.info(f'User: {user_input}')
                logger.info('------------ Answer ------------')
                logger.info(f'Role: {target}')
            
            logger.info('Program terminated due to user exit.')
            exit()


        # test mode
        if args.language == 'zh':
            inputs = ['你好，我是林临。', \
                    '你有什么兴趣爱好吗？', \
                    '最近过得怎么样？']
        else:
            inputs = ['Hi, I\'m Lin.',\
                    'Do you have any hobbies or interests?', \
                    'How is your day going on?']

        for i in inputs:
            
            processed_input = f'{i}{extra_prompt}'
            target = model.chat(processed_input)

            logger.info('----------- Question -----------')
            logger.info(f'User: {i}')
            logger.info('------------ Answer ------------')
            logger.info(f'Role: {target}')

        
    except Exception as e:
        progress.stop()

        logger.exception(e)
        logger.error('Program terminated with error.')


    
        





