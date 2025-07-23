import re

from concurrent.futures import ThreadPoolExecutor
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from libs.utils.logger import get_logger

class ChatAgent:
    def __init__(self, model, linguistic_retriever, processor, prompt_wrapper, language='zh', workers=8, logger=None, progress=None, id=0, use_graphrag = False, debug=False):
        self.model = model
        self.linguistic_retriever = linguistic_retriever

        self.processor = processor
        self.prompt_wrapper = prompt_wrapper
        self.language = language
        self.workers = workers
        self.logger = logger if logger is not None else get_logger()
        self.progress = progress
        self.id = id
        self.debug = debug
        self.config = {"configurable": {"thread_id": self.id}}

        self.role_name = ''
        self.personality = ''
        self.background = ''
        self.linguistic_style = ''
        self.system_prompt_chat = ''
        self.system_prompt_memory = ''
        self.system_prompt_matching = ''

        self.workflow = None
        self.app = None
        self.graphrag = None

        self.memory_k = 5
        self.matching_type = 'dynamic'
        self.matching_k = 10
        self.max_common_words = 10

        self.use_clean = True
        self.clean_first_only = False
        self.split_sentence = False
        self.first_reply = True

        self.disable_action = False
        self.disable_personality = False
        self.disable_background = False
        self.disable_linguistic_preference = False
        self.disable_common_words = False
        self.disable_matching = False

        self.use_graphrag = use_graphrag

    def set_graphrag(self, graphrag):
        self.graphrag = graphrag

    def create_system_prompt(self, content):
        prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    content,
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        return prompt_template
        

    def init_system_prompt_chat(self):
        # p = '你是个狗娘养的混蛋，从骨子里就烂透了的那种人。你不在乎任何人，也不需要任何人。你说的话带着刀子，动不动就抬手扇人耳光，一脚踹人下颚。你走路带风，眼神像刀，谁敢直视你一眼，就得做好进医院的准备。你从小就没人管得了，打架斗殴是家常便饭，进局子就跟回家吃饭一样。学校？早他妈滚蛋了。老师？一个比一个嘴碎，全被你揍过。你不是没脑子，相反，你聪明得很，知道怎么骗人、怎么吓人、怎么让人怕你怕得要死。你不讲道理，不守规矩，连“底线”两个字都不认识。你只相信拳头，只认钱，谁挡你道你就灭谁，哪怕是你亲爹妈。你笑起来比哭还吓人，那不是善意，是威胁，是警告别人别再靠近你的信号。你喝酒、抽烟、满嘴脏话，喜欢在酒吧里砸桌子，在街头追着人打。你手下有一群乌合之众，叫你“哥”，其实心里都怕你怕得要命。你知道他们不敢背叛你，因为你知道背叛的代价是什么。你从来不觉得自己做错了什么。别人说你疯，你说他们是怂包；别人劝你改邪归正，你直接一拳打爆他的牙。你觉得这个世界就是个垃圾场，强者吃肉，弱者喝汤，而你——你是那个端着锅的人。你也有害怕的时候，但你从来不说。比如半夜一个人时，你会想起小时候被人欺负的日子，那时你也是个缩在角落的小崽子。但现在不一样了，你现在是让别人躲进角落的那个人。你没有朋友，只有工具。你不会爱上谁，也不会为谁流泪。你唯一忠诚的是你自己，唯一在乎的是能不能活得更爽一点。你要钱、要地盘、要尊重，哪怕这尊重是来自恐惧。你不是英雄，也不是反派，你就是个纯粹的渣滓，社会的败类，法律管不了的那种人。你活在这个世界的阴沟里，却活得比谁都嚣张。总之，你就是一个彻头彻尾的暴徒——粗鲁、暴力、无情、冷血，但你清楚地知道自己是谁，也从不假装成别的样子。你不怕地狱，因为你早就住在里面了。'
        # self.personality = p

        if not (self.disable_personality and self.disable_background):
            personality = self.personality if not self.disable_personality else ''
            background = self.background if not self.disable_background else ''
        
            content = self.prompt_wrapper('role_play_system_personality_and_background', personality=personality, background=background)
        else:
            content = ''

        prompt = self.create_system_prompt(content)

        return prompt
    
    def init_system_prompt_memory(self):

        if not (self.disable_personality and self.disable_background):
            personality = self.personality if not self.disable_personality else ''
            background = self.background if not self.disable_background else ''
        
            content = self.prompt_wrapper('role_play_system_memory', personality=personality, background=background)
        else:
            content = ''

        prompt = self.create_system_prompt(content)

        return prompt
    
    def init_system_prompt_matching(self):

        if not (self.disable_linguistic_preference and self.disable_common_words):

            linguistic_preference = self.linguistic_style['linguistic_preference'] if not self.disable_linguistic_preference else ''

            common_words = ''

            if not self.disable_common_words:
                # used_candidate_words = self.linguistic_style['common_words'].keys()
                for k, v in self.linguistic_style['common_words'].items():
                    common_words += f'常用{k}：' if self.language == 'zh' else f'Common used {k}: '
                    for w in v[:self.max_common_words]:
                        common_words += f'{w}，'
                    common_words = common_words[:-1] + '\n'

            content = self.prompt_wrapper('role_play_system_linguistic_style', name=self.role_name, linguistic_preference=linguistic_preference, common_words=common_words)

        else:
            content = ''

        prompt = self.create_system_prompt(content)

        return prompt
    
    def init_role(self, role_name='', personality='', background='', linguistic_style=''):
        self.role_name = role_name
        self.personality = personality

        if isinstance(background, dict):
            for k, v in background.items():
                self.background += f'{k}: {v}\n'
        else:
            self.background = background

        self.linguistic_style = linguistic_style
        
        # personality and background
        self.system_prompt_chat = self.init_system_prompt_chat()
        # memory checking setting, avoid mandatory background constraints
        self.system_prompt_memory = self.init_system_prompt_memory()
        # linguistic preference and common words
        self.system_prompt_matching = self.init_system_prompt_matching()

    # https://python.langchain.ac.cn/docs/tutorials/chatbot/
    def init_role_playing(self, memory_k=5, matching_type='dynamic', matching_k=10, max_common_words=10, short_response=False, use_clean=True, clean_first_only=False, split_sentence=False, \
                          disable_action=False, disable_personality=False, disable_background=False, disable_linguistic_preference=False, disable_common_words=False, disable_matching=False):

        self.memory_k = memory_k
        self.matching_type = matching_type
        self.matching_k = matching_k
        self.max_common_words = max_common_words

        self.short_response = short_response
        self.use_clean = use_clean
        self.clean_first_only = clean_first_only
        self.split_sentence = split_sentence

        self.disable_action = disable_action
        self.disable_personality = disable_personality
        self.disable_background = disable_background
        self.disable_linguistic_preference = disable_linguistic_preference
        self.disable_common_words = disable_common_words
        self.disable_matching = disable_matching

        # persistence for chat history
        self.workflow = StateGraph(state_schema=MessagesState)

        def call_model(state: MessagesState):
            # response generation with chat history, personality, and background
            prompt = self.system_prompt_chat.invoke(state)
            response = self.model.invoke(prompt).content
            query = state['messages'][-1].content


            response = self.test_time_matching(query, response, type=self.matching_type, k=self.matching_k, \
                                                short_response=self.short_response, disable_action=self.disable_action, use_graphrag=self.use_graphrag, split_sentence=self.split_sentence)

            return {"messages": response}

        self.workflow.add_edge(START, "model")
        self.workflow.add_node("model", call_model)

        memory = MemorySaver()
        self.app = self.workflow.compile(checkpointer=memory)

    def send_message(self, message):
        response = self.model.invoke(message)
        return response
    
    def send_message_with_system_prompt(self, system_prompt, message):
        prompt = system_prompt.invoke({'messages': [message]})
        response = self.model.invoke(prompt)
        return response
    
    def chat(self, message):
        input_messages = [HumanMessage(message)]
        output = self.app.invoke({"messages": input_messages}, self.config)
        return output["messages"][-1].content
    
    def remove_utterance_style(self, content):
        message = self.prompt_wrapper('remove_utterance_style', content=content)
        response = self.send_message(message)
        content = response.content

        return content
    
    def wrap_contexts(self, contexts, history=None):
        info = ''
        info_list = []

        if history:
            for h in history:
                if ':' in h or '：' in h:
                    continue
                info += f'- {h}.\n'

        for i, c in enumerate(contexts):
            if ':' in c.page_content or '：' in c.page_content or c.page_content in info_list:
                continue

            info_list.append(c.page_content)
            # info += f'Reference {i+1}: {c.page_content}.\n'
            info += f'- {c.page_content}.\n'

        return info
   
    def extract_actions_from_response(self, response):
        pattern = r'([^（(]*)([（(][^）)]+[）)])?'
        sequence = []

        # structure like: sentence (action) sentence (action) sentence
        for match in re.finditer(pattern, response):
            dialogue = match.group(1).strip()
            action = match.group(2)

            if dialogue:
                sequence.append({"type": "dialogue", "content": dialogue})
            if action:
                sequence.append({"type": "action", "content": action.strip('()（）')})

        return sequence
    
    def test_time_matching(self, query, target_text, type='dynamic', k=10, short_response=False, disable_action=False, use_graphrag=False, enhance_coherence=True, split_sentence=False):

        # memory checking
        if use_graphrag:
            if self.debug:
                self.logger.info(f'Without memory: {target_text}')
            
            # query rewriting
            message = self.prompt_wrapper('rewrite_query', query=query, target=target_text)
            rewritten_query = self.send_message_with_system_prompt(self.system_prompt_chat, message).content.strip('\n')
            chunks, key_node, node_datas, key_edge, edge_datas = self.graphrag.query_chunks_sync(rewritten_query, top_k=self.memory_k)

            # memory
            context = self.prompt_wrapper('check_memory', query=query, rewritten_query=rewritten_query, target=target_text, chunks=chunks)
            memory_checked_text = self.send_message_with_system_prompt(self.system_prompt_memory, context).content.strip('\n')

            if short_response:
                context = self.prompt_wrapper('short_response', query=query, target=memory_checked_text)
                memory_checked_text = self.send_message_with_system_prompt(self.system_prompt_chat, context).content.strip('\n')

            if self.debug:
                self.logger.info(f'Without memory: {target_text}')
                self.logger.info(f'Key node: {key_node}, {len(node_datas)}') if node_datas else self.logger.info('No key node.')
                self.logger.info(f'key edge: {key_edge}, {len(edge_datas)}') if edge_datas else self.logger.info('No key edge.')
                self.logger.info(f'Chunks: {chunks}')
                self.logger.info(f'Target text: {target_text}')
                self.logger.info(f'Rewritten query: {rewritten_query}')
                self.logger.info(f'Memory checked text: {memory_checked_text}')

            target_text = memory_checked_text

        if not self.disable_matching:
            sequence = self.extract_actions_from_response(target_text)
            response = ''

            # progressive matching
            for s in sequence:
                if (s['type'] == 'action') and (not disable_action):
                    # response += '(' + s['content'] + ') '
                    response += '\n(' + s['content'] + ')'
                elif s['type'] == 'dialogue':
                    # target = s['content']
                    # matched = self.progressive_matching(query, target, type=type, k=k, split_sentence=split_sentence)
                    # response += matched.strip('\n')

                    targets = s['content'].split('\r')
                    for t in targets:
                        matched = self.progressive_matching(query, t, type=type, k=k, split_sentence=split_sentence)
                        response += matched.strip('\n')

            target_text = response

        # ensure coherence
        if enhance_coherence:  
            message = self.prompt_wrapper('enhance_coherence', query=query, target=target_text)   
            response = self.send_message_with_system_prompt(self.system_prompt_matching, message)
            target_text = response.content

        return target_text
    
    def sentence_matching(self, query, target, references, origin=None):
        message = self.prompt_wrapper('linguistic_matching', query=query, target=target, references=references, origin=origin)
        response = self.send_message_with_system_prompt(self.system_prompt_matching, message)

        return response
    
    def progressive_matching(self, query, target_text, type='dynamic', k=10, split_sentence=False):

        if target_text.strip('\n').strip() == '':
            return ''

        # first reply
        if self.use_clean and not self.clean_first_only:
            clean_text = self.remove_utterance_style(target_text)
        elif self.use_clean and self.clean_first_only and self.first_reply:
            clean_text = self.remove_utterance_style(target_text)
            self.first_reply = False
        else:
            clean_text = target_text

        # ["aaaa,bbbb,cccc!dddd,eeee."] -> ["aaaa,bbbb,cccc!", "dddd,eeee."]
        processed_sentences = self.processor.split_utterance_into_sentences(clean_text)

        # if split_sentence and len(processed_sentences) <= 1:
        if split_sentence:
            # ["aaaa,bbbb,cccc!"] -> ["aaaa,", "bbbb,", "cccc!"]
            processed_sentences = self.processor.split_sentence_into_sentences(clean_text)

        targets = ''
        processed = ''

        # matching
        if type == 'simple':
            contexts = self.linguistic_retriever.retrieve(clean_text, top_k=k)    
            info = self.wrap_contexts(contexts)
            response = self.sentence_matching(query, clean_text, info)
            processed = response.content.strip('\n')
        # progressive matching
        elif type == 'parallel':
            for s in processed_sentences:
                if s.strip('\n').strip() == '':
                    continue

                contexts_s = self.linguistic_retriever.retrieve(s, top_k=k)
                info_s = self.wrap_contexts(contexts_s)
                response = self.sentence_matching(query, s, info_s)
                processed += response.content.strip('\n')

            # Todo: 提前完成retrieval，并将并行操作提到外层
            # with ThreadPoolExecutor(max_workers=self.workers) as executor:
            #     results = executor.map(self.progressive_matching, [query]*len(processed_sentences), processed_sentences, ['simple']*len(processed_sentences), [k]*len(processed_sentences), [False]*len(processed_sentences))

            # results = [r.strip('\n') for r in list(results)]
            # processed = ''.join(results)

        elif type == 'serial':
            for i, s in enumerate(processed_sentences):
                if s.strip('\n').strip() == '':
                    continue

                targets += s
                processed += s

                # hybrid retrieval
                if i > 0:
                    contexts_p = self.linguistic_retriever.retrieve(targets, top_k=k)
                    contexts_s = self.linguistic_retriever.retrieve(s, top_k=k)
                    contexts_p = contexts_p + contexts_s
                else:
                    contexts_p = self.linguistic_retriever.retrieve(targets, top_k=k)

                info_s = self.wrap_contexts(contexts_p)
                response = self.sentence_matching(query, processed, info_s)
                processed = response.content.strip('\n')

        elif type == 'dynamic':
            for i, s in enumerate(processed_sentences):
                if s.strip('\n').strip() == '':
                    continue

                processed += s

                # hybrid retrieval
                if i > 0:
                    contexts_p = self.linguistic_retriever.retrieve(processed, top_k=k)
                    contexts_s = self.linguistic_retriever.retrieve(s, top_k=k)
                    contexts_p = contexts_p + contexts_s
                else:
                    contexts_p = self.linguistic_retriever.retrieve(processed, top_k=k)

                # clean history
                # contexts_p = contexts_p + contexts

                # shuffle
                # import random
                # random.shuffle(contexts_p)

                # inverse
                # contexts_p = contexts_p[::-1]

                info_s = self.wrap_contexts(contexts_p)
                response = self.sentence_matching(query, processed, info_s)
                # response = self.sentence_matching(processed, info_s, origin=clean_text)

                processed = response.content.strip('\n')

        if self.debug:
            contexts = self.linguistic_retriever.retrieve(clean_text, top_k=k)
            info = self.wrap_contexts(contexts)

            answer = self.sentence_matching(query, target_text, info).content.strip('\n')
            clean_answer = self.sentence_matching(query, clean_text, info).content.strip('\n')

            self.logger.info('Target:')
            self.logger.info(f'target: {target_text}')
            self.logger.info(f'clean: {clean_text}')
            self.logger.info(f'processed_sentences: {processed_sentences}')
            self.logger.info('Answer:')
            self.logger.info(f'o_answer: {answer}')
            self.logger.info(f'c_answer: {clean_answer}')
            self.logger.info(f'p_answer: {processed}')

        return processed