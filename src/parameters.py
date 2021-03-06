import argparse
import sys, json
from distutils.util import strtobool

class Params:
    def __init__(self):
        parser = argparse.ArgumentParser(description='Preprocess')
        parser.add_argument('-debug', action='store', default=False, type=strtobool)
        parser.add_argument('-bert_name', action='store', default='bert-base-uncased', type=str)
        parser.add_argument('-word_embedding_dropout', action='store', default=0.05, type=float)
        parser.add_argument('-cuda_devices', action='store', default='0', type=str)
        parser.add_argument('-allen_lazyload', action='store', default=True, type=strtobool)
        parser.add_argument('-batch_size_for_train', action='store', default=32, type=int)
        parser.add_argument('-batch_size_for_eval', action='store', default=8, type=int)
        parser.add_argument('-batch_size_for_kb_encoder', action='store', default=5000, type=int)
        parser.add_argument('-hard_negatives_num', action='store', default=10, type=int)
        parser.add_argument('-num_epochs', action='store', default=3, type=int)
        parser.add_argument('-lr', action="store", default=1e-5, type=float)
        parser.add_argument('-weight_decay', action="store", default=0, type=float)
        parser.add_argument('-beta1', action="store", default=0.9, type=float)
        parser.add_argument('-beta2', action="store", default=0.999, type=float)
        parser.add_argument('-epsilon', action="store", default=1e-8, type=float)
        parser.add_argument('-amsgrad', action='store', default=False, type=strtobool)
        # After split with BERTtokenizer
        parser.add_argument('-max_title_len', action='store', default=12, type=int)
        parser.add_argument('-max_desc_len', action='store', default=50, type=int)
        parser.add_argument('-max_context_len_after_tokenize', action='store', default=100, type=int)

        parser.add_argument('-add_mse_for_biencoder', action='store', default=False, type=strtobool)
        parser.add_argument('-search_method', action='store', default='indexflatip', type=str)
        parser.add_argument('-add_hard_negatives', action='store', default=True, type=strtobool)
        parser.add_argument('-metionPooling', action='store', default="CLS", type=str)
        parser.add_argument('-entityPooling', action='store', default="CLS", type=str)
    
        parser.add_argument('-dimentionReduction', action='store', default=False, type=strtobool)
        parser.add_argument('-dimentionReductionToThisDim', action='store', default=300, type=int)

        # Custom arguments

        ## Load from checkpoint
        parser.add_argument('-load_from_checkpoint', action='store', default=False, type=strtobool)
        parser.add_argument('-model_path', action='store', default='./', type=str)
        parser.add_argument('-mention_encoder_filename', action='store', default='mention_encoder.bin', type=str)
        parser.add_argument('-entity_encoder_filename', action='store', default='entity_encoder.bin', type=str)
        parser.add_argument('-model_filename', action='store', default='model.bin', type=str)

        ## Save to checkpoint
        parser.add_argument('-save_checkpoints', action='store', default=False, type=strtobool)

        ## Entities
        parser.add_argument('-save_entities', action='store', default=False, type=strtobool)
        parser.add_argument('-load_entities', action='store', default=False, type=strtobool)
        parser.add_argument('-entities_path', action='store', default='/home/repositories/Zero-Shot-Entity-Linking/src', type=str)
        parser.add_argument('-entities_filename', action='store', default='entities.t7', type=str)

        parser = self.fixed_params_for_preprocess_adder(parser=parser)
        self.opts = parser.parse_args(sys.argv[1:])

        print('\n===PARAMETERS===')
        for arg in vars(self.opts):
            print(arg, getattr(self.opts, arg))
        print('===PARAMETERS END===\n')

    def get_params(self):
        return self.opts

    def fixed_params_for_preprocess_adder(self, parser):
        parser.add_argument('-extracted_first_token_for_description', action='store', default=100, type=int)
        parser.add_argument('-extracted_first_token_for_title', action='store', default=16, type=int)

        parser.add_argument('-dataset_dir', action='store', default='./data/', type=str)
        parser.add_argument('-documents_dir', action='store', default='./data/documents/', type=str)
        parser.add_argument('-mentions_dir', action='store', default='./data/mentions/', type=str)
        parser.add_argument('-mentions_splitbyworld_dir', action='store', default='./data/mentions_split_by_world/', type=str)
        parser.add_argument('-mention_leftandright_tokenwindowwidth', action='store', default=40, type=int)
        parser.add_argument('-debugSampleNum', action='store', default=100000000, type=int)
        parser.add_argument('-dir_for_each_world', action='store', default='./data/worlds/', type=str)
        parser.add_argument('-experiment_logdir', action='store', default='./src/experiment_logdir/', type=str)
        return parser

    def dump_params(self, experiment_dir):
        parameters = vars(self.opts)
        with open(experiment_dir + 'parameters.json', 'w') as f:
            json.dump(parameters, f, ensure_ascii=False, indent=4, sort_keys=False, separators=(',', ': '))

