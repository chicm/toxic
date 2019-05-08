import json

def replace_words(text: str, mapping: dict):
    """
    Replaces unusual punctuation with normal.

    :param text: text to clean
    :param mapping: dict with mapping
    :return: cleaned text
    """
    for word in mapping:
        if word in text:
            text = text.replace(word, mapping[word])

    return text

def load_preprocessing_data():
    """
    Loads dict with various mappings and strings for cleaning.

    :return:
    """
        
    with open('mapping_dict.json', 'r') as f:
        mapping_dict = json.load(f)

    # combine several dicts into one
    replace_dict = {
        #**mapping_dict['contraction_mapping'],
        **mapping_dict['mispell_dict'],
        #**mapping_dict['special_punc_mappings'],
        **mapping_dict['rare_words_mapping'],
        #**mapping_dict['bad_case_words'],
        **mapping_dict['mis_spell_mapping']
    }

    mapping_dict = {
        #'spaces': mapping_dict['spaces'],
        #'punctuation': mapping_dict['punctuation'],
        'words_to_replace': replace_dict
    }

    return mapping_dict

mapping_dict = load_preprocessing_data()

def clean_special_chars(text, punct):
    for p in punct:
        text = text.replace(p, ' ')
    return text

def _preprocess(text):
    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
    #CHARS_TO_REMOVE = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n“”’\'∞θ÷α•à−β∅³π‘₹´°£€\×™√²—'

    result = replace_words(text, mapping_dict['words_to_replace'])
    result = clean_special_chars(result, punct)

    return result

def preprocess_text(data):
    '''
    Credit goes to https://www.kaggle.com/gpreda/jigsaw-fast-compact-solution
    '''


    data = data.astype(str).apply(lambda x: _preprocess(x))

    return data

if __name__ == '__main__':
    import pandas as pd
    import os
    import settings
    df = pd.read_csv(os.path.join(settings.DATA_DIR, 'train.csv'))
    df['counts'] = df.comment_text.map(lambda x: sum([ 1 if k in x else 0 for k in mapping_dict['words_to_replace']]))
    total = df.counts.values.sum()
    print('total occur:', total)