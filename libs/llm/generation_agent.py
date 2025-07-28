import re

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from libs.utils.logger import get_logger

class GenerationAgent():
    def __init__(self, model, prompt_wrapper, language='zh', workers=8, logger=None, progress=None, debug=False):
        self.model = model
        self.prompt_wrapper = prompt_wrapper
        self.language = language
        self.workers = workers
        self.logger = logger if logger is not None else get_logger()
        self.progress = progress
        self.step_progress = None
        self.debug = debug

    def send_message(self, message):
        response = self.model.invoke(message)
        return response
    
    def summarize_from_chunk(self, chunk, skip=False):
        if skip:
            chunk.metadata['summary'] = 'No summary available.'
            if self.progress is not None and self.step_progress is not None:
                self.progress.advance(self.step_progress) 
            return chunk
        
        message = self.prompt_wrapper('summarize_chunks', content=chunk.page_content)
        response = self.send_message(message)
        content = response.content
        chunk.metadata['summary'] = content

        if self.progress is not None and self.step_progress is not None:
            self.progress.advance(self.step_progress) 
        return chunk

    def summarize_from_chunks_wrapper(self, chunks, skip=False, serial=False):
        results = []

        self.step_progress = self.progress.add_task(description=f"Summarizing from {len(chunks)} chunks", total=len(chunks))

        if serial:
            for c in chunks:
                r = self.summarize_from_chunk(c, skip=skip)
                results.append(r)
        else:
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                results = executor.map(self.summarize_from_chunk, chunks, [skip]*len(chunks))
        
        chunks = list(results)

        if not self.debug:
            self.progress.update(self.step_progress, visible=False)
            self.progress.refresh()
        self.step_progress = None

        return chunks

    def extract_info_from_content(self, content):
        lines = content.split('\n')
        info_dict = {}
        for line in lines:
            line = line.strip()

            if not line:
                continue

            try:
                key, value = re.split(r'[:：]', line, 1)
            except:
                key, value = None, None

            if key and value:
                info_dict[key.strip()] = value.strip()

        return info_dict
    
    def extract_dialogues(self, chunk):
        message = self.prompt_wrapper('extract_dialogues', content=chunk['context'])
        response = self.send_message(message)
        content = response.content
        info = self.extract_info_from_content(content)

        if self.progress is not None and self.step_progress is not None:
            self.progress.advance(self.step_progress) 
        return info

    def extract_dialogues_wrapper(self, chunks, serial=False):
        dialogues = defaultdict(set)
        results = []

        self.step_progress = self.progress.add_task(description=f"Extracting dialogues from {len(chunks)} chunks", total=len(chunks))

        if serial:
            for c in chunks:
                info = self.extract_dialogues(c)
                results.append(info)
        else:
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                results = executor.map(self.extract_dialogues, chunks)
        
        for info in results:
            for k, v in info.items():
                dialogues[k].add(v.replace('\"', ''))

        dialogues = {k: list(v) for k, v in dialogues.items()}

        if not self.debug:
            self.progress.update(self.step_progress, visible=False)
            self.progress.refresh()
        self.step_progress = None

        return dialogues
    
    def detect_role_entities(self, candidate):
        message = self.prompt_wrapper('detect_role_entity', content=candidate)
        response = self.send_message(message)
        content = response.content.strip()

        if self.progress is not None and self.step_progress is not None:
            self.progress.advance(self.step_progress) 
        return content
    
    def detect_role_entities_wrapper(self, candidates, serial=False):
        entities = []
        non_entities = []
        results = []

        self.step_progress = self.progress.add_task(description=f"Detecting role entities from {len(candidates)} candidates", total=len(candidates))

        if serial:
            for c in candidates:
                content = self.detect_role_entities(c)
                results.append(content)
        else:
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                results = executor.map(self.detect_role_entities, candidates)

        for c, content in zip(candidates, results):

            if ('否' in content) or ('No' in content):
                non_entities.append(c)
                continue

            entities.append(c)
        
        if not self.debug:
            self.progress.update(self.step_progress, visible=False)
            self.progress.refresh()
        self.step_progress = None

        return entities, non_entities

    def combine_roles_from_entities(self, entities, max_name_length=10):
        roles = set()
        mapping = {}
        indexes = {}

        indexes['failed'] = []

        task = self.progress.add_task(description=f"Combining roles from {len(entities)} entities", total=len(entities))

        for e in entities:
            message = self.prompt_wrapper('connect_entity_with_roles', entity=e, roles=roles)
            response = self.send_message(message)
            content = response.content.strip()

            self.progress.advance(task)

            if ('否' in content) or ('No' in content):
                name_parts = re.split(r'[，,]', content)
                if len(name_parts) > 1:
                    name = name_parts[1].strip()
                    if len(name) <= max_name_length:
                        roles.add(name)
                        mapping[e] = name
                        indexes.setdefault(name, []).append(e)
                        continue
    
                mapping[e] = 'failed'
                indexes['failed'].append(e)
            else:
                if len(content) <= max_name_length:
                    roles.add(content)
                    mapping[e] = content
                    indexes.setdefault(content, []).append(e)
                else:
                    mapping[e] = 'failed'
                    indexes['failed'].append(e)

        if not self.debug:
            self.progress.update(task, visible=False)
            self.progress.refresh()

        roles = list(roles)

        return roles, mapping, indexes
    

    def get_related_chunks(self, chunk, roles):
        message = self.prompt_wrapper('get_related_chunks', chunk=chunk, role=roles)
        response = self.send_message(message)
        content = response.content.strip()

        if self.progress is not None and self.step_progress is not None:
            self.progress.advance(self.step_progress) 
        return content
    
    def get_related_chunks_wrapper(self, chunks, roles, serial=False):
        related_chunks_indexes = []
        results = []

        self.step_progress = self.progress.add_task(description=f"Getting related chunks of {roles} from {len(chunks)} chunks", total=len(chunks))

        if serial:
            for i, c in enumerate(chunks):
                content = self.get_related_chunks(c, roles)
                results.append(content)
        else:
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                results = executor.map(self.get_related_chunks, chunks, [roles]*len(chunks))

        for i, content in enumerate(results):
            if ('否' in content) or ('No' in content):
                continue

            related_chunks_indexes.append(i)

        if not self.debug:
            self.progress.update(self.step_progress, visible=False)
            self.progress.refresh()
        self.step_progress = None

        return related_chunks_indexes

    def analyze_personality_from_chunks(self, chunk, roles):
        message = self.prompt_wrapper('analyze_personality_from_chunk', chunk=chunk, role=roles)
        response = self.send_message(message)
        content = response.content.strip()

        if self.progress is not None and self.step_progress is not None:
            self.progress.advance(self.step_progress) 
        return content
    
    def analyze_personality_from_chunks_wrapper(self, chunks, roles, serial=False):
        characters = set()
        used_related_chunks = []
        results = []

        self.step_progress = self.progress.add_task(description=f"Analyzing personality of {roles} from {len(chunks)} chunks", total=len(chunks))
        
        if serial:
            for i, c in enumerate(chunks):
                content = self.analyze_personality_from_chunks(c, roles)
                # content_parts = [p for p in content.split(" ") if p.strip()]
                results.append(content)
        else:
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                results = executor.map(self.analyze_personality_from_chunks, chunks, [roles]*len(chunks))

        for i, content in enumerate(results):
            if 'FALSE' in content:
                continue

            content_parts = [p for p in content.split(" ") if p.strip()]            
            characters.update(content_parts)
            used_related_chunks.append(i)

        characters = list(characters)

        message = self.prompt_wrapper('summarize_personality', content=characters)
        response = self.send_message(message)
        content = response.content.strip()

        if not self.debug:
            self.progress.update(self.step_progress, visible=False)
            self.progress.refresh()
        self.step_progress = None

        return content, characters, used_related_chunks
    
    def extract_background_from_chunk(self, chunk, role, candidate_keys):
        message = self.prompt_wrapper('extract_background_from_chunk', chunk=chunk, role=role, keys=candidate_keys)
        response = self.send_message(message)
        content = response.content.strip()
        return content
    
    def combine_background_key(self, background, key):
        post_message = self.prompt_wrapper('combine_duplicate_backgrounds', content=background)
        response = self.send_message(post_message)
        content = response.content.strip().replace('/n', '')
        
        split_content = re.split(r'[；;、]', content)
        temp = set(bg.strip() for bg in split_content if bg.strip())
        content = ';'.join(temp)

        return content, key

    def combine_backgrounds(self, background, candidate_keys):
        new_background = {}
        for k in background:
            if k in candidate_keys:
                new_background[k], _ = self.combine_background_key(background[k], k)
                
        return new_background
    
    def process_background_from_chunks(self, chunks, role, candidate_keys, bad_info):
        background = {}

        for i, c in enumerate(chunks):
            content = self.extract_background_from_chunk(c, role, candidate_keys)

            if content == 'FALSE':
                continue

            info = self.extract_info_from_content(content)
            for k in info:
                k_search = k.strip().lower()
                info[k] = info[k].strip()
                if k_search not in background:
                    background[k_search] = info[k]
                    continue

                bad = any(b in info[k] and len(info[k]) < len(background[k_search]) for b in bad_info)
                if not bad:
                    background[k_search] = background[k_search] + ';' + info[k]

        background = self.combine_backgrounds(background, candidate_keys)

        self.progress.advance(self.step_progress, len(chunks))
        
        return background
    
    def get_info_for_background_extraction(self):
        if self.language == 'zh':
            candidate_keys = ['姓名', '性别', '年龄', '种族','身份职业', '外貌特征', '身体状况', '家庭情况', '时代背景', '人际关系', '重要财产物品', '特殊技能', '兴趣爱好'] 
        else:
            candidate_keys = ['gender', 'name', 'age', 'race', 'occupation', 'appearance', 'physical condition', 'family background', 'historical era', 'interpersonal relationship', 'important possessions', 'skills', 'hobbies']
        
        bad_info = ['提及', '信息', '未知', 'FALSE', 'information', 'provided', 'specified']
        
        return candidate_keys, bad_info

    def extract_background_from_chunks(self, chunks, roles, freq=10):
        background = {}

        candidate_keys, bad_info = self.get_info_for_background_extraction()
        
        self.step_progress = self.progress.add_task(description=f"Extracting background of {roles} from {len(chunks)} chunks", total=len(chunks))
        combination_task = self.progress.add_task(description=f"Combining background of {roles} from {len(chunks)} chunks", total=len(chunks))

        def worker(chunk_batch):
            try:
                partial_background = self.process_background_from_chunks(chunk_batch, roles, candidate_keys, bad_info)
            except Exception as e:
                self.logger.error(f'Error processing background from chunk batch: {e}')

            return partial_background

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = []
            for i in range(0, len(chunks), freq):
                chunk_batch = chunks[i:i+freq]
                futures.append(executor.submit(worker, chunk_batch))

            def combine_background_mp(background, candidate_keys):
                new_background = {}
                combined_futures = []
                
                for k in background:
                    if k in candidate_keys:
                        combined_futures.append(executor.submit(self.combine_background_key, background[k], k))

                for future in as_completed(combined_futures):
                    new_background[future.result()[1]] = future.result()[0]

                return new_background

            count = 0
            for future in as_completed(futures):
                count += 1
                partial_background = future.result()
                for k in partial_background:
                    if k not in background:
                        background[k] = partial_background[k]
                    else:
                        split_existing = set(re.split(r'[；;、]', background[k]))
                        split_new = set(re.split(r'[；;、]', partial_background[k]))
                        combined = ';'.join(split_existing.union(split_new))
                        background[k] = combined

                if count % freq == 0:
                    background = combine_background_mp(background, candidate_keys)

                    if self.debug:
                        self.logger.info('----------- Background -----------')
                        for k in background:
                            self.logger.info(f'{k}: {background[k]}')

                    self.progress.advance(combination_task, freq * freq)

        combined_background = combine_background_mp(background, candidate_keys)
        self.progress.advance(combination_task, len(chunks)%(freq * freq))

        if not self.debug:
            self.progress.update(self.step_progress, visible=False)
            self.progress.update(combination_task, visible=False)
            self.progress.refresh()

        return combined_background
    
    def extract_background_from_chunks_serial(self, chunks, roles, freq=5):
        background = {}

        candidate_keys, bad_info = self.get_info_for_background_extraction()

        task = self.progress.add_task(description=f"Extracting background of {roles} from {len(chunks)} chunks", total=len(chunks))

        count = 0

        for i, c in enumerate(chunks):
            content = self.extract_background_from_chunk(c, roles, candidate_keys)

            self.progress.advance(task)

            if not ('FALSE' == content):
                count += 1

                info = self.extract_info_from_content(content)

                for k in info:
                    k_search = k.strip().lower()
                    info[k] = info[k].strip()
                    if k_search not in background:
                        background[k_search] = info[k]
                    else:
                        bad = False
                        for b in bad_info:
                            if b in info[k] and len(info[k]) < len(background[k_search]):
                                bad = True
                        if not bad:
                            background[k_search] = background[k_search] + ';' + info[k]
            
            if ((count % freq == 0) and (not ('FALSE' == content))) or (i == len(chunks) - 1):
                new_background = {}
                for k in background:
                    if k in candidate_keys:
                        post_message = self.prompt_wrapper('combine_duplicate_backgrounds', content=background[k])
                        response = self.send_message(post_message)
                        content = response.content.strip().replace('/n', '')

                        split_content = re.split(r'[；;、]', content)
                        temp = set(bg.strip() for bg in split_content if bg.strip())
                        content = ';'.join(temp) 
                        new_background[k] = content

                background = new_background


                if self.debug:
                    self.logger.info('----------- Background -----------')
                    for k in background:
                        self.logger.info(f'{k}: {background[k]}')


        if not self.debug:
            self.progress.update(task, visible=False)
            self.progress.refresh()

        return background
    
    def analyze_linguistic_style_from_sentences(self, sentences, max_words=10):

        words_history = {}
        words_count = defaultdict(int)

        # Extract part-of-speech from sentences
        if self.language == 'zh':
            from jieba import posseg
            from libs.utils.static import get_jieba_paddle_pos_map

            pos_map = get_jieba_paddle_pos_map()
            
            for s in sentences:
                # words = posseg.cut(s)

                # paddle mode
                words = posseg.lcut(s, use_paddle=True)

                for w in words:
                    if w.flag in pos_map:
                        words_history.setdefault(pos_map[w.flag], set()).add(w.word)
                        words_count[w.word, 0] += 1
        else:
            import nltk 
            from libs.utils.static import get_nltk_pos_map_en, get_wordnet_pos

            pos_map = get_nltk_pos_map_en()

            nltk.download('punkt_tab')
            nltk.download('averaged_perceptron_tagger_eng')
            nltk.download('wordnet')

            lemmatizer = nltk.stem.WordNetLemmatizer()

            def tokenize_and_lemmatize(sentence):

                tokens = nltk.word_tokenize(sentence)
                tagged_tokens = nltk.pos_tag(tokens)

                results = []

                for word, tag in tagged_tokens:
                    wntag = get_wordnet_pos(tag)

                    if wntag:
                        lemma = lemmatizer.lemmatize(word, pos=wntag)
                        used_pos = wntag
                    else:
                        lemma = lemmatizer.lemmatize(word)
                        if tag.startswith('PRP'):
                            tag = 'p'  # pronoun
                        used_pos = tag

                    results.append({
                        'word': word,
                        'pos': tag,
                        'used_pos': used_pos,
                        'lemma': lemma
                    })

                return results
            
            for s in sentences:

                words = tokenize_and_lemmatize(s)

                for w in words:
                    if w['used_pos'] in pos_map:
                        words_history.setdefault(pos_map[w['used_pos']], set()).add(w['lemma'])
                        words_count[w['lemma'], 0] += 1

        # Sort and select top words
        for k in words_history:
            words_history[k] = sorted(words_history[k], key=lambda x: words_count[x, 0], reverse=True)
            # words_history[k] = words_history[k][:max_words]

        message = self.prompt_wrapper('analyze_linguistic_style', content=sentences)
        response = self.send_message(message)
        content = response.content.strip()

        return {'linguistic_preference': content, 'common_words': words_history}
