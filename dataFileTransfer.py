# csv
# :source
# text1, text2
# :target
# text1 (wakati), label-num
# text2 (wakati), label-num
# note: remove \u3000 (caps Space),  http? , #something, etc...

from __future__ import print_function
from __future__ import division

import random

import MeCab as mecab
import re
import csv
import glob

""" 日本語の解析のためのユーティリティ """
ws = re.compile(' ')
wakati_tagger = mecab.Tagger('-Owakati')
hiragana_tagger = mecab.Tagger('-Oyomi')


def get_word_list(sentence):
    """
    Notice: returned list will not contain '、'.
    ['今日', 'は', '良い', '天気', 'です', 'ね', '。']
    :param sentence: simple text
    :return: list of word
    """
    return [x for x in ws.split(wakati_tagger.parse(sentence))
            if x not in ['"', '\ufeff', '、', '”']][:-1]


def get_hiragana_list(sentence):
    """
    Notice: returned list will not contain '、'.
    get_word_yomi_list('今日は、良い天気ですね。')
    ['キ', 'ョ', 'ウ', 'ハ', 'ヨ', 'イ', 'テ', 'ン', 'キ', 'デ', 'ス', 'ネ', '。']
    :param sentence: simple text
    :return: list of hiragana
    """
    return [x for x in hiragana_tagger.parse(sentence)
            if x not in ['"', '\ufeff', '、', '”']][:-1]


def get_word_yomi_list(sentence):
    """
    Notice: returned list will not contain '、'.
    Example: get_word_yomi_list('今日は、良い天気ですね。')
    ['キョウ', 'ハ', 'ヨイ', 'テンキ', 'デス', 'ネ', '。']
    :param sentence: simple text
    :return: list of HIRAGANA word
    """
    return [hiragana_tagger.parse(x).rstrip()
            for x in ws.split(wakati_tagger.parse(sentence))
            if x not in ['"', '\ufeff', '、', '”']][:-1]


def read_from_csv(path, parse_func):
    lists = []
    with open(path, 'r', encoding='utf-8') as f:
        sources = csv.reader(f, delimiter=',')
        for start, end in sources:
            if start not in lists:
                lists += [start]
            if end not in lists:
                lists += [end]
    lists = [parse_func(l.replace('"', '').replace('～～～', '～')) for l in lists]
    return lists


def read_from_csvs(head_path, parse_func):
    path_lists = glob.glob(head_path, recursive=False)
    corpus = []
    for p in path_lists:
        corpus += read_from_csv(p, parse_func=parse_func)
    return corpus


def replace_lists(text, lists):
    for m in lists:
        text = text.replace(m[0], m[1])
    return text


if __name__ == '__main__':
    path1 = 'japanese-corpus/usually/'
    one_file1 = '1.csv'
    every_file = '*'
    corpus1 = read_from_csvs(path1 + every_file, get_word_list)
    import pprint

    # pprint.pprint(corpus1)
    path2 = 'or-corpus/'
    corpus2 = read_from_csvs(path2 + every_file, get_word_list)
    # pprint.pprint(corpus2)

    # アド', 'モス', '商会
    # オー', 'ガス
    # '～', '～', '～' -> '～'
    #  '・', '・', '・' -> '・・・'
    # remove '"', '\ufeff'
    fil = [['アド モス 商会', 'アドモス商会'],
           ['オー ガス', 'オーガス'],
           ['～ ～ ～', '～'],
           ['・ ・ ・', '…'],
           ['!…', '！ …']]

    corpus1 = ['"' + replace_lists(' '.join(item), fil) + '",1' for item in corpus1]
    corpus2 = ['"' + replace_lists(' '.join(item), fil) + '",2' for item in corpus2]
    with open('train.csv', 'w', encoding='utf-8') as f:
        for line in corpus1 + corpus2:
            f.write(line + '\n')
    with open('test.csv', 'w', encoding='utf-8') as f:
        for line in corpus1 + corpus2:
            if random.random() > 0.8:
                f.write(line + '\n')
