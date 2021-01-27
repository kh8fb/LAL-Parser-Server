"""
Loads and runs the parser for LAL-parsing
"""

import argparse
import collections
import itertools
import os.path
import time

import torch
import torch.optim.lr_scheduler

from ..KM_parser import ChartParser, use_cuda, BERT_TOKEN_MAPPING
import nltk
from nltk import word_tokenize, sent_tokenize
from tqdm import tqdm
from .vocabulary import vocabulary

REVERSE_TOKEN_MAPPING = dict([(value, key) for key, value in BERT_TOKEN_MAPPING.items()])

def load_model(model_path, cuda):
    """
    Loads the model onto the designated device.

    Parameters
    ----------
    model_path: str
        Path to the saved LAL Parser .pth file.
    cuda: bool
        Determines whether or not to load the model onto a GPU device.
    Returns
    -------
    model: torch.model
        Model with the preloaded weights.
    """
    if use_cuda:
        return torch.load(model_path)
    else:
        return torch.load(model_path, map_location=lambda storage, location: storage)

    if cuda:
        if use_cuda:
            return torch.load(model_path)
        else:
            raise AttributeError("Cuda flag was specified but no CUDA device available")
    else:
        if use_cuda:
            # must run this in order to properly load model
            pass
        return torch.load(model_path, map_location=lambda storage, location: storage)


def run_parser(info, sequences, dependency, constituency):
    """
    Run's the model on each of the sentences in a list of sequences.

    Parameters
    ----------
    info: torch.model
        LAL-Parser model with the preloaded weights.
    sequences: dict
        Dictionary with key strings for the "id" and value strings for the
        sentences that are meant to be parsed.
    dependency: bool
        Defines whether or not to return dependency parsing results in final dictionary
    constituency: bool
        Defines whether or not to return constituency parsing results in final dictionary
    Returns
    -------
    return_dict: dict
        Dictionary with each entry containing a string "id" as the key and a
        parsed tree string of the sentence as a the value.
    """
    assert 'hparams' in info['spec'], "Older savefiles not supported"
    parser = ChartParser.from_spec(info['spec'], info['state_dict'])
    parser.eval()
    print("Parsing sentences...")

    ids, sentences = zip(*sequences.items())
    sentences = [sentence.strip() for sentence in sentences if len(sentence.strip()) > 0]

    pos_tag = 1
    eval_batch_size = 50

    syntree_pred = []
    for start_index in tqdm(range(0, len(sentences), eval_batch_size), desc='Parsing sentences'):
        subbatch_sentences = sentences[start_index:start_index+eval_batch_size]
        if pos_tag == 2:
            tagged_sentences = [[(dummy_tag, REVERSE_TOKEN_MAPPING.get(word, word)) for word in word_tokenize(sentence)] for sentence in subbatch_sentences]
        elif pos_tag == 1:
            tagged_sentences = [[(REVERSE_TOKEN_MAPPING.get(tag, tag), REVERSE_TOKEN_MAPPING.get(word, word)) for word, tag in nltk.pos_tag(word_tokenize(sentence))] for sentence in subbatch_sentences]
        else:
            tagged_sentences = [[(REVERSE_TOKEN_MAPPING.get(word.split('_')[0],word.split('_')[0]), REVERSE_TOKEN_MAPPING.get(word.split('_')[1],word.split('_')[1])) for word in sentence.split()] for sentence in subbatch_sentences]
        syntree, _ = parser.parse_batch(tagged_sentences)
        syntree_pred.extend(syntree)
    
    return_dict = {}
    depend_heads = [[int(leaf.father) for leaf in tree.leaves()] for tree in syntree_pred]
    constituencies = [tree.convert().linearize() for tree in syntree_pred]
    depend_labels = [[leaf.type for leaf in tree.leaves()] for tree in syntree_pred]
    if constituency:
        return_dict["constituencies"] = dict(zip(ids, constituencies))
    if dependency:
        return_dict["dependency labels"] = dict(zip(ids, depend_labels))
        return_dict["dependency heads"] = dict(zip(ids, depend_heads))
    return return_dict
