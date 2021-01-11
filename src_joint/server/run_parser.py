"""
Loads and runs the parser for LAL-parsing
"""

import argparse
import itertools
import os.path
import time

import torch
import torch.optim.lr_scheduler

from ..KM_parser import ChartParser
import nltk
from nltk import word_tokenize, sent_tokenize
from tqdm import tqdm


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
    if cuda:
        if torch.cuda.is_available():
            return torch.load(load_path)
        else:
            raise AttributeError("Cuda flag was specified but no CUDA device available")
    else:
        return torch.load(load_path, map_location=lambda storage, location: storage)


def run_parser(info, sequences):
    """
    Run's the model on each of the sentences in a list of sequences.

    Parameters
    ----------
    info: torch.model
        LAL-Parser model with the preloaded weights.
    sequences: dict
        Dictionary with key strings for the "id" and value strings for the
        sentences that are meant to be parsed.
    Returns
    -------
    return_dict: dict
        Dictionary with each entry containing a string "id" as the key and a
        parsed tree string of the sentence as a the value.
    """
    assert 'hparams' in info['spec'], "Older savefiles not supported"
    parser = KM_parser.ChartParser.from_spec(info['spec'], info['state_dict'])
    parser.eval()
    print("Parsing sentences...")

    ids, sentences = zip(*sequences.items())
    sentences = [sentence.strip() for sentence in sentences if len(sentence.strip()) > 0]

    pos_tag = 1
    eval_batch_size = 50

    """def save_data(syntree_pred, cun):
        pred_head = [[leaf.father for leaf in tree.leaves()] for tree in syntree_pred]
        pred_type = [[leaf.type for leaf in tree.leaves()] for tree in syntree_pred]
        appent_string = "_" + str(cun) + ".txt"
        if args.output_path_synconst != '-':
            with open(args.output_path_synconst + appent_string, 'w') as output_file:
                for tree in syntree_pred:
                    output_file.write("{}\n".format(tree.convert().linearize()))
            print("Output written to:", args.output_path_synconst)

        if args.output_path_syndep != '-':
            with open(args.output_path_syndep + appent_string, 'w') as output_file:
                for heads in pred_head:
                    output_file.write("{}\n".format(heads))
            print("Output written to:", args.output_path_syndep)

        if args.output_path_synlabel != '-':
            with open(args.output_path_synlabel + appent_string, 'w') as output_file:
                for labels in pred_type:
                    output_file.write("{}\n".format(labels))
            print("Output written to:", args.output_path_synlabel)"""

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
    
    parsed_sentences = [tree.convert().linearize() for tree in syntree_pred]
    return_dict = dict(zip(ids, parsed_sentences))
    return returned_dict
