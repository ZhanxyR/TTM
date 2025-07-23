
def get_jieba_paddle_pos_map():
    pos_map =  {
        'n': '名词',
        'f': '方位词',
        's': '处所词',
        't': '时间词',
        'v': '动词',
        'vd': '动副词',
        'vn': '名动词',
        'a': '形容词',
        'ad': '副形词',
        'an': '名形词',
        'd': '副词',
        'm': '数词',
        'q': '量词',
        'r': '代词',
        'p': '介词',
        'c': '连词',
        'u': '助词',
        'xc': '其他虚词',
    }
    return pos_map


def get_nltk_pos_map_en():
    pos_map = {
        'a': 'adjective',
        'n': 'noun',
        'v': 'verb',
        'r': 'pronoun',
        'p': 'personal pronoun',
        'MD': 'modal',
        'IN': 'preposition',
    }
    return pos_map

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return 'a'  # adjective
    elif treebank_tag.startswith('V'):
        return 'v'  # verb
    elif treebank_tag.startswith('N'):
        return 'n'  # noun
    elif treebank_tag.startswith('R'):
        return 'r'  # adverb
    else:
        return None 