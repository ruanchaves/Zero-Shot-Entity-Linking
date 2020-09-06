# Biencoder imports
import pdb, time
from utils import experiment_logger, cuda_device_parser, dev_or_test_finallog, dev_or_test_finallog_rawdata, worlds_loader
from parameters import Params
from data_reader import WorldsReader
import torch
import numpy as np
from embeddings import EmbLoader
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer
from encoders import Pooler_for_mention, Pooler_for_title_and_desc
from model import Biencoder
import torch.optim as optim
from evaluator import oneLineLoaderForDevOrTestEvaluation, oneLineLoaderForDevOrTestEvaluationRawData, devEvalExperimentEntireDevWorldLog, devEvalExperimentEntireDevWorldLogRawData
from token_indexing import TokenIndexerReturner
from hardnegative_searcher import HardNegativesSearcherForEachEpochStart

# Custom imports
import json
from utils import load_model_objects, save_model_objects
import os, pathlib
from evaluator import Evaluate_one_world_raw_data
import copy 
from data_reader import WorldsReaderOnline
from argparse import Namespace

# Flask
from flask import Flask, jsonify, request
app = Flask(__name__)

# Torch variables
torch.backends.cudnn.deterministic = True
seed = 777
np.random.seed(seed)
torch.manual_seed(seed)

api_endpoint = "/api"
parameters_json = '/home/repositories/Zero-Shot-Entity-Linking/src/parameters.json'

def load_opts_from_json(fname):
    with open(fname,'r') as f:
        dct = json.load(f)
    return Namespace(**dct)

def main():
    print("Loading biencoder.")
    opts = load_opts_from_json(parameters_json)
    cuda_devices = cuda_device_parser(str_ids=opts.cuda_devices)
    TRAIN_WORLDS, DEV_WORLDS, TEST_WORLDS = worlds_loader(args=opts)

    vocab = Vocabulary()
    iterator_for_training_and_evaluating_mentions = BucketIterator(batch_size=opts.batch_size_for_train,
                                                                   sorting_keys=[('context', 'num_tokens')])
    iterator_for_training_and_evaluating_mentions.index_with(vocab)

    embloader = EmbLoader(args=opts)
    emb_mapper, emb_dim, textfieldEmbedder = embloader.emb_returner()
    tokenIndexing = TokenIndexerReturner(args=opts)
    global_tokenizer = tokenIndexing.berttokenizer_returner()
    global_tokenIndexer = tokenIndexing.token_indexer_returner()

    if opts.load_from_checkpoint:
        mention_encoder, entity_encoder, model = load_model_objects(
            model_path=opts.model_path,
            mention_encoder_filename=opts.mention_encoder_filename,
            entity_encoder_filename=opts.entity_encoder_filename,
            model_filename=opts.model_filename)
        mention_encoder.share_memory()
        entity_encoder.share_memory()
        model.share_memory()
    else:
        mention_encoder = Pooler_for_mention(args=opts, word_embedder=textfieldEmbedder)
        entity_encoder = Pooler_for_title_and_desc(args=opts, word_embedder=textfieldEmbedder)
        model = Biencoder(args=opts, mention_encoder=mention_encoder, entity_encoder=entity_encoder, vocab=vocab)
    model = model.cuda()

    with torch.no_grad():
        finalEvalFlag = 0
        world_name = 'wikipedia'
        dev_or_test_flag = 'test'

        reader_for_eval = WorldsReaderOnline(args=opts, world_name=world_name, token_indexers=global_tokenIndexer,
                                        tokenizer=global_tokenizer)
        Evaluator = Evaluate_one_world_raw_data(args=opts, world_name=world_name,
                                        reader=reader_for_eval,
                                        embedder=textfieldEmbedder,
                                        trainfinished_mention_encoder=mention_encoder,
                                        trainfinished_entity_encoder=entity_encoder,
                                        vocab=vocab, experiment_logdir=None,
                                        dev_or_test=dev_or_test_flag,
                                        berttokenizer=global_tokenizer,
                                        bertindexer=global_tokenIndexer)
        Evaluate_one_world_raw_data.finalEvalFlag = copy.copy(finalEvalFlag)

        return Evaluator

def predict(message):
    Evaluator.enable_logging = False
    with torch.no_grad():
        trainEpoch = -1
        how_many_top_hits_preserved = 1
        Evaluator.set_mentions(message)
        result = Evaluator.evaluate_one_world(trainEpoch=trainEpoch, how_many_top_hits_preserved=how_many_top_hits_preserved)
        for dct in result:
            result_ids_dict = {}
            for k, v in dct.items():
                result_ids_dict[k] = v[0]
            yield result_ids_dict

Evaluator = main()

@app.route('/', methods=['GET', 'POST'])
def model_entrypoint():
    with open(parameters_json,'r') as f:
        parameters = json.load(f)
    content = request.get_json()
    res = []
    for item in predict(content):
        res.append(item)
    response = {
        "response": res,
        "parameters": parameters
    }
    return jsonify(response)