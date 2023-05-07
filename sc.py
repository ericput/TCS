import os
import math
import logging
import yaml
import csv
import argparse
import pickle
import random
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm, trange
from tensorboardX import SummaryWriter
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class BertForSC(BertPreTrainedModel):

    def __init__(self, model_config, num_labels, dropout_rate):
        super(BertForSC, self).__init__(model_config)
        self.bert = BertModel(model_config)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.hidden2label = torch.nn.Linear(model_config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, segment_ids, input_mask, label_id=None):
        repre, _ = self.bert(input_ids, segment_ids, attention_mask=input_mask, output_all_encoded_layers=False)
        repre = self.dropout(repre)
        # repre, _ = torch.max(repre, dim=1)
        repre = torch.mean(repre, dim=1)
        logits = self.hidden2label(repre)
        if label_id is not None:
            loss = F.cross_entropy(logits, label_id)
            return loss
        else:
            return F.softmax(logits, -1)


class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class InputFeatures(object):

    def __init__(self, input_ids, segment_ids, input_mask, label_id=None):
        self.input_ids = input_ids
        self.segment_ids = segment_ids
        self.input_mask = input_mask
        self.label_id = label_id


class DataProcessor(object):

    @staticmethod
    def read_csv(input_file, quotechar=None):
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    @staticmethod
    def get_examples(data_file, set_type):
        raise NotImplementedError()

    @staticmethod
    def get_labels():
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

class CLSCProcessor(DataProcessor):

    @staticmethod
    def get_examples(data_file, set_type):
        lines = DataProcessor.read_csv(data_file)
        examples = []
        for index, item in enumerate(lines):
            guid = '%s-%d' % (set_type, index)
            examples.append(InputExample(guid=guid, text_a=item[1], text_b=item[2], label=item[0]))
        return examples

    @staticmethod
    def get_labels():
        return ['0', '1']

class MLDocProcessor(DataProcessor):

    @staticmethod
    def get_examples(data_file, set_type):
        examples = []
        if isinstance(data_file, str):
            for index, line in enumerate(open(data_file, 'r', encoding='utf-8')):
                guid = '%s-%d' % (set_type, index)
                label, content = line.strip().split('\t')
                examples.append(InputExample(guid=guid, text_a=eval(content).decode('utf-8'), text_b=None, label=label))
        elif isinstance(data_file, list):
            for index, (content_line, label_line) in enumerate(zip(open(data_file[0], encoding='utf-8'),
                                                                   open(data_file[1], encoding='utf-8'))):
                guid = '%s-%d' % (set_type, index)
                content = content_line.strip()
                label = label_line.strip()
                examples.append(InputExample(guid=guid, text_a=content, text_b=None, label=label))
        return examples

    @staticmethod
    def get_labels():
        return ['CCAT', 'ECAT', 'GCAT', 'MCAT']

data_processors = {'clsc': CLSCProcessor, 'mldoc': MLDocProcessor}

task_metrics = {'clsc': accuracy_score, 'mldoc': accuracy_score}

def convert_examples_to_features(examples):
    """Loads a data file into a list of `InputBatch`s."""

    def _truncate_seq_pair(tokens_a, tokens_b, max_length):
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]
        tokens = ['[CLS]'] + tokens_a + ['[SEP]']
        segment_ids = [0] * len(tokens)
        if tokens_b:
            tokens += tokens_b + ['[SEP]']
            segment_ids += [1] * (len(tokens_b) + 1)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        if example.label is not None:
            label_id = label_map[example.label]
        else:
            label_id = None
        features.append(InputFeatures(input_ids=input_ids, segment_ids=segment_ids, input_mask=input_mask, label_id=label_id))
    return features


class SCDataset(Dataset):
    def __init__(self, examples, training=True):
        features = convert_examples_to_features(examples)
        if training:
            self.tensors = [[torch.LongTensor(f.input_ids), torch.LongTensor(f.segment_ids),
                             torch.LongTensor(f.input_mask), torch.LongTensor([f.label_id])] for f in features]
        else:
            self.tensors = [[torch.LongTensor(f.input_ids), torch.LongTensor(f.segment_ids),
                             torch.LongTensor(f.input_mask)] for f in features]
            if features[0].label_id is None:
                self.label_ids = None
            else:
                self.label_ids = [item.label_id for item in features]


    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, item):
        return self.tensors[item]


def inference(infer_model, data, batch_size, device):
    sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    with torch.no_grad():
        confidences, predictions = [], []
        for batch in tqdm(dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, segment_ids, input_mask = batch
            batch_dists = infer_model(input_ids, segment_ids, input_mask)
            batch_confidences, batch_predictions = torch.max(batch_dists, -1)
            confidences.extend(batch_confidences.tolist())
            predictions.extend(batch_predictions.tolist())
    return predictions, confidences


def main():

    global model

    if config['train']['do']:

        if config['task']['frozen_level'] > 0:
            for n, p in model.named_parameters():
                if 'embedding' in n:
                    p.requires_grad = False
                elif 'encoder' in n:
                    if config['task']['frozen_level'] > 1:
                        if int(n.split('.')[3]) < 3:
                            p.requires_grad = False

        model.train()

        train_examples = processor.get_examples(data_file=config['train']['data_file'], set_type='train')
        train_data = SCDataset(train_examples, training=True)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=config['train']['batch_size'])
        total_train_steps = math.ceil(len(train_data) / config['train']['batch_size']) * config['train']['epochs']

        dev_data = SCDataset(processor.get_examples(data_file=config['eval']['data_file'], set_type='dev'), training=False)

        logger.info('***** Running training *****')
        logger.info('  Num examples = %d', len(train_data))
        logger.info('  Batch size = %d', config['train']['batch_size'])
        logger.info('  Num steps = %d', total_train_steps)

        all_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        no_decay = ['bias', 'LayerNorm.weight']
        grouped_parameters = [
            {'params': [p for n, p in all_params if not any(nd in n for nd in no_decay)], 'weight_decay': config['train']['weight_decay']},
            {'params': [p for n, p in all_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = BertAdam(grouped_parameters, lr=config['train']['learning_rate'], warmup=config['train']['warmup_proportion'], t_total=total_train_steps)
        summary = SummaryWriter(os.path.join(config['task']['output_dir'], 'runs'))
        global_step = 0
        dev_score = {}
        least_save_score = 0
        for _ in trange(0, config['train']['epochs'], desc='Epoch'):
            for step, batch in enumerate(tqdm(train_dataloader, desc='Iteration')):
                batch = tuple(t.to(device) for t in batch)
                input_ids, segment_ids, input_mask, label_id = batch
                loss = model(input_ids, segment_ids, input_mask, label_id.squeeze(-1))
                loss = loss.mean()
                summary.add_scalar('loss', loss, global_step)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                global_step += 1
                if global_step > config['eval']['begin'] and global_step % config['eval']['interval'] == 0:
                    model.eval()
                    predictions, _ = inference(model, dev_data, config['eval']['batch_size'], device)
                    model.train()
                    score = metrics(dev_data.label_ids, predictions)
                    summary.add_scalar('dev_score', score, global_step)
                    save_ckpt = False
                    remove_ckpt = None
                    if score > least_save_score:
                        save_ckpt = True
                        dev_score[global_step] = score
                        if len(dev_score) > config['eval']['max_save']:
                            remove_ckpt = sorted(dev_score.items(), key=lambda x:x[1])[0][0]
                            least_save_score = dev_score.pop(remove_ckpt)
                    if save_ckpt:
                        model_to_save = model.module if hasattr(model, 'module') else model
                        torch.save({'model_state': model_to_save.state_dict(), 'max_seq_length': max_seq_length, 'lower_case': config['task']['lower_case']},
                                   os.path.join(config['task']['output_dir'], 'checkpoint-{}'.format(global_step)))
                    if remove_ckpt is not None:
                        os.system('rm {}'.format(os.path.join(config['task']['output_dir'], 'checkpoint-{}'.format(remove_ckpt))))
        dev_score = sorted(dev_score.items(), key=lambda x:x[1], reverse=True)
        logger.info(dev_score)
        with open(os.path.join(config['task']['output_dir'], 'dev_score.txt'), 'w', encoding='utf-8') as f:
            f.write(str(dev_score)+'\n')
        best_ckpt = os.path.join(config['task']['output_dir'], 'checkpoint-{}'.format(dev_score[0][0]))
        logging.info('Load %s' % best_ckpt)
        best_checkpoint = torch.load(best_ckpt, map_location='cpu')
        model = BertForSC.from_pretrained(config['task']['bert_model_dir'], state_dict=best_checkpoint['model_state'],
                                          num_labels=len(label_map), dropout_rate=config['train']['dropout_rate'])
        model.to(device)

    if config['predict']['do']:

        predict_data = SCDataset(processor.get_examples(data_file=config['predict']['data_file'], set_type='predict'), training=False)
        model.eval()
        predictions, confidences = inference(model, predict_data, config['predict']['batch_size'], device)
        if predict_data.label_ids is not None:
            score = metrics(predict_data.label_ids, predictions)
            logger.info(score)
            with open(os.path.join(config['task']['output_dir'], 'metrics.txt'), 'w', encoding='utf-8') as writer:
                writer.write(str(score))
        with open(os.path.join(config['task']['output_dir'], 'predictions.txt'), 'w', encoding='utf-8') as writer:
            for item in predictions:
                writer.write(str(item) + '\n')
        with open(os.path.join(config['task']['output_dir'], 'confidences.bin'), 'wb') as handle:
            pickle.dump(confidences, handle)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default=None)
    parser.add_argument('--config_bin', type=str, default=None)
    args = parser.parse_args()
    if args.config_file:
        if os.path.exists(args.config_file):
            with open(args.config_file) as cf:
                config = yaml.load(cf.read())
        else:
            print('Config file not exists')
            exit(1)
    elif args.config_bin:
        if os.path.exists(args.config_bin):
            with open(args.config_bin, 'rb') as handle:
                config = pickle.load(handle)
        else:
            print('Config file not exists')
            exit(1)
    else:
        print('No config information.')
        exit(1)

    os.makedirs(config['task']['output_dir'], exist_ok=True)

    if not os.path.exists(os.path.join(config['task']['output_dir'], 'config.yaml')):
        with open(os.path.join(config['task']['output_dir'], 'config.txt'), 'w') as writer:
            writer.write(str(config))

    random.seed(config['train']['seed'])
    np.random.seed(config['train']['seed'])
    torch.manual_seed(config['train']['seed'])
    if config['use_cuda'] and torch.cuda.is_available():
        device = torch.device('cuda', torch.cuda.current_device())
        torch.cuda.manual_seed_all(config['train']['seed'])
    else:
        device = torch.device('cpu')

    max_seq_length = config['task']['max_seq_length']
    metrics = task_metrics[config['task']['task_name']]
    processor = data_processors[config['task']['task_name']]()
    label_map = {label: i for i, label in enumerate(processor.get_labels())}
    logger.info(label_map)
    tokenizer = BertTokenizer.from_pretrained(config['task']['bert_model_dir'],
                                              do_lower_case=config['task']['lower_case'])
    if config['task']['checkpoint']:
        logging.info('Load %s' % config['task']['checkpoint'])
        checkpoint = torch.load(config['task']['checkpoint'], map_location='cpu')
        model = BertForSC.from_pretrained(config['task']['bert_model_dir'], state_dict=checkpoint['model_state'], num_labels=len(label_map),
                                          dropout_rate=config['train']['dropout_rate'],)
    else:
        logger.info('Load %s' % config['task']['bert_model_dir'])
        model = BertForSC.from_pretrained(config['task']['bert_model_dir'], num_labels=len(label_map), dropout_rate=config['train']['dropout_rate'])

    model.to(device)

    main()
