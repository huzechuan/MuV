from pathlib import Path
import re
from data_preprocess.utils import configparser

class Params:
    def __init__(self, args, method):
        super(Params, self).__init__()
        config_file = Path(f'./config/{method}') / Path(args.config)
        print(config_file)
        config = configparser()
        config.read(config_file, encoding='utf-8')
        self.config = config
        self.args = args

        self.task = config.get('Task', 'task')
        self.metric = config.get('Task', 'metric')

        self.freeze = config.getboolean('Model', 'freeze')
        self.save_model = config.get('Model', 'save_model')

        self.dataset = config.get('Data', 'dataset')
        self.model_path = Path('./models') / self.dataset / 'Direct'

        self.BERT = config.get('Embedding', 'BERT')

        # Hyper-parameter
        self.batch_size = config.getint('Hyper', 'batch_size')
        self.max_epoch = config.getint('Hyper', 'max_epoch')
        self.HP_L2 = config.getfloat('Hyper', 'HP_L2')
        self.HP_BERT_lr = config.getfloat('Hyper', 'HP_BERT_lr')
        self.percent_of_labeled_data = config.getfloat('Hyper', 'percent_of_labeled_data')
        self.percent_of_unlabeled_data = config.getfloat('Hyper', 'percent_of_unlabeled_data')

        self.language = args.language#  config.get('Data', 'language')  #
        self.name = args.name#config.get('Data', 'name')  #


class Direct(Params):
    def __init__(self, args):
        self.method = 'Direct'
        super(Direct, self).__init__(args, self.method)
        self.save_model = self.config.getboolean('Model', 'save_model')
        self.mode = self.config.get('Model', 'mode')

        self.BERT = self.config.get('Embedding', 'BERT')

        self.batch_size = self.config.getint('Hyper', 'batch_size')
        self.max_epoch = self.config.getint('Hyper', 'max_epoch')
        self.HP_L2 = self.config.getfloat('Hyper', 'HP_L2')
        self.HP_BERT_lr = self.config.getfloat('Hyper', 'HP_BERT_lr')

        target_language = args.target_language#config.get('Data', 'target_language')
        self.target_languages = re.split(',', target_language)  # target data
        target_name = args.target_name#config.get('Data', 'target_name')
        self.target_names = re.split(',', target_name)  # target name


class KD(Params):
    def __init__(self, config):
        super(KD, self).__init__(config)
        self.config = config
        self.method = 'KD'

        self.freeze = config.getboolean('Model', 'freeze')
        self.consensus = config.get('Model', 'consensus')
        self.result_path = Path('./results/KD') / self.data

        self.interpolation = config.getfloat('Hyper', 'interpolation')


class Concat(Params):
    def __init__(self, config):
        super(Concat, self).__init__(config)
        self.config = config
        self.method = 'Concat'

        self.freeze = config.getboolean('Model', 'freeze')
        self.consensus = config.get('Model', 'consensus')
        self.result_path = Path('./results/Concat') / self.data

        self.interpolation = config.getfloat('Hyper', 'interpolation')

        self.rounds = config.getint('Others', 'rounds')


class Vote(Params):
    def __init__(self, config):
        super(Vote, self).__init__(config)
        self.config = config
        self.method = 'Vote'

        self.freeze = config.getboolean('Model', 'freeze')
        self.consensus = config.get('Model', 'consensus')
        self.result_path = Path('./results/Vote') / self.data

        self.interpolation = config.getfloat('Hyper', 'interpolation')


class MView(Params):
    def __init__(self, args):
        self.method = 'mview'
        super(MView, self).__init__(args, self.method)

        self.freeze = self.config.getboolean('Model', 'freeze')
        self.consensus = self.config.get('Model', 'consensus')

        self.HP_lr = self.config.getfloat('Hyper', 'HP_lr')
        self.interpolation = self.config.getfloat('Hyper', 'interpolation')
        self.view_interpolation = self.config.getfloat('Hyper', 'view_interpolation')
        self.mu = self.config.getfloat('Hyper', 'mu')

        self.att_dropout = self.config.getfloat('Hyper', 'att_dropout')
        self.sample_rate = self.config.getfloat('Hyper', 'sample_rate')

        self.source_language = re.split(',', args.source_name)
        self.aggregate_method = args.aggregate_method


class Semi(Params):
    def __init__(self, config):
        super(Semi, self).__init__(config)
        self.config = config
        self.method = 'Semi'

        self.result_path = Path('./results/Semi') / self.data
        self.add_nsample = config.getint('Hyper', 'add_nsample')
        self.rounds = config.getint('Others', 'rounds')
        self.source_data = [Path(self.dataset) / source for source in self.source_language]
        self.source_name = re.split(',', config.get('Data', 'source_language'))
        self.langs = [self.name] + self.source_name
