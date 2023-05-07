import os
import math
import logging
import yaml
import random
import pickle
import argparse
from span_f1 import SpanBasedF1Measure
from tqdm import tqdm, trange
from tensorboardX import SummaryWriter
import numpy as np
import torch
from torch.nn.functional import softmax
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class BertForSL(BertPreTrainedModel):

    def __init__(self, model_config, num_labels, dropout_rate):
        super(BertForSL, self).__init__(model_config)
        self.num_labels = num_labels
        self.bert = BertModel(model_config)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.hidden2label = torch.nn.Linear(model_config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, input_mask, predict_mask=None, one_hot_labels=None):
        repre, _ = self.bert(input_ids, attention_mask=input_mask, output_all_encoded_layers=False)
        repre = self.dropout(repre)
        logits = self.hidden2label(repre)
        p = softmax(logits, -1)
        if one_hot_labels is not None:
            losses = -torch.sum(one_hot_labels*torch.log(p), -1)
            losses = torch.masked_select(losses, predict_mask)
            return torch.sum(losses)
        else:
            return p


class InputExample(object):

    def __init__(self, guid, units, labels=None):
        self.guid = guid
        self.units = units
        self.labels = labels

class InputFeatures(object):

    def __init__(self, input_ids, input_mask, predict_mask=None, one_hot_labels=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.predict_mask = predict_mask
        self.one_hot_labels = one_hot_labels

class DataProcessor(object):

    @staticmethod
    def create_examples_from_conll_format_file(data_file, set_type='data'):
        examples, units, labels = [], [], []
        index = 0
        for line in open(data_file, encoding='utf-8'):
            segs = line.split()
            if not segs:
                guid = '%s-%d' % (set_type, index)
                index += 1
                examples.append(InputExample(guid=guid, units=units, labels=labels))
                units = []
                labels = []
            else:
                units.append(segs[0])
                if segs[-1] in label_map:
                    labels.append(segs[-1])
                else:
                    labels.append('O')
        if units:
            guid = '%s-%d' % (set_type, index)
            examples.append(InputExample(guid=guid, units=units, labels=labels))
        return examples

    @staticmethod
    def create_examples_from_files(content_file, label_file=None, set_type='data'):
        examples = []
        index = 0
        if label_file is not None:
            for content_line, label_line in zip(open(content_file, encoding='utf-8'),
                                                open(label_file, encoding='utf-8')):
                guid = '%s-%d' % (set_type, index)
                index += 1
                units = content_line.split()
                labels = label_line.split()
                for i in range(len(labels)):
                    if labels[i] not in label_map:
                        labels[i] = 'O'
                examples.append(InputExample(guid=guid, units=units, labels=labels))

        else:
            for content_line in open(content_file, encoding='utf-8'):
                guid = '%s-%d' % (set_type, index)
                index += 1
                units = content_line.split()
                examples.append(InputExample(guid=guid, units=units))
        return examples

    @staticmethod
    def get_examples(data_file, set_type):
        if isinstance(data_file, str):
            return DataProcessor.create_examples_from_conll_format_file(data_file, set_type)
        elif isinstance(data_file, list):
            if len(data_file) == 2:
                content_file, label_file = data_file
            else:
                content_file = data_file[0]
                label_file = None
            return DataProcessor.create_examples_from_files(content_file, label_file, set_type)

    @staticmethod
    def get_labels():
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

class CONLLProcessor(DataProcessor):

    @staticmethod
    def get_labels():
        return ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']

class MSRAProcessor(DataProcessor):

    @staticmethod
    def get_labels():
        return ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']


processors = {
        'conll': CONLLProcessor,
        'msra': MSRAProcessor
}

def convert_examples_to_features(examples, training):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    line_tags = []
    for (ex_index, example) in enumerate(examples):
        samples = []
        context, tokens, predict_mask, label_ids = [], [], [], []
        if example.labels:
            labels = example.labels
        else:
            labels = ['O'] * len(example.units)
        for i, w in enumerate(example.units):
            if w == '[MASK]':
                sub_words = ['[MASK]']
            else:
                sub_words = tokenizer.tokenize(w)
                if not sub_words:
                    sub_words = ['[UNK]']
            tokens.extend(sub_words)
            predict_mask.append(1)
            predict_mask.extend([0] * (len(sub_words) - 1))
            label_ids.append(label_map[labels[i]])
            label_ids.extend([0] * (len(sub_words) - 1))
            while len(context) + len(tokens) >= max_seq_length - 2:
                l = max_seq_length - len(context) - 2
                samples.append([['[CLS]'] + context + tokens[:l] + ['[SEP]'], [0] * (len(context) + 1) + predict_mask[:l] + [0], [0] * (len(context) + 1) + label_ids[:l] + [0]])
                if not context:
                    line_tags.append(1)
                else:
                    line_tags.append(0)
                context = tokens[max(0, l - max_seq_length // 2):l]
                tokens, predict_mask, label_ids = tokens[l:], predict_mask[l:], label_ids[l:]
        if sum(predict_mask):
            samples.append([['[CLS]'] + context + tokens + ['[SEP]'], [0] * (len(context) + 1) + predict_mask + [0], [0] * (len(context) + 1) + label_ids + [0]])
            if not context:
                line_tags.append(1)
            else:
                line_tags.append(0)

        for s in samples:
            input_ids = tokenizer.convert_tokens_to_ids(s[0])
            input_mask = [1] * len(input_ids)
            padding_length = max_seq_length - len(input_ids)
            zero_padding = [0] * padding_length
            input_ids += zero_padding
            input_mask += zero_padding
            predict_mask = s[1] + zero_padding
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(predict_mask) == max_seq_length
            if training and example.labels:
                label_ids = s[2] + zero_padding
                assert len(label_ids) == max_seq_length
                one_hot_labels = np.eye(len(label_map), dtype=np.float32)[label_ids]
            else:
                one_hot_labels = None
            features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask, predict_mask=predict_mask, one_hot_labels=one_hot_labels))
    assert len(examples) == sum(line_tags), logger.error('{} != {}'.format(len(examples), sum(line_tags)))
    return features, line_tags


class SLDataset(Dataset):
    def __init__(self, examples, training=True):
        features, line_tags = convert_examples_to_features(examples, training)
        if training:
            self.tensors = [[torch.LongTensor(f.input_ids), torch.LongTensor(f.input_mask),
                             torch.BoolTensor(f.predict_mask), torch.FloatTensor(f.one_hot_labels)] for f in features]
        else:
            self.features = features
            self.line_tags = line_tags
            if examples[0].labels is None:
                self.labels = None
            else:
                self.labels = [ex.labels for ex in examples]
            self.tensors = [[torch.LongTensor(f.input_ids), torch.LongTensor(f.input_mask)] for f in features]

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
            input_ids, input_mask = batch
            batch_dists = infer_model(input_ids, input_mask)
            batch_confidences, batch_predictions = torch.max(batch_dists, -1)
            confidences.extend(batch_confidences.tolist())
            predictions.extend(batch_predictions.tolist())
    predictions_p, pp = [], []
    confidences_p, cp = [], []
    for i in range(len(predictions)):
        if data.line_tags[i] and pp:
            predictions_p.append(pp)
            pp = []
            confidences_p.append(cp)
            cp = []
        feature = data.features[i]
        for j in range(sum(feature.input_mask)):
            if feature.predict_mask[j] == 1:
                pp.append(label_list[predictions[i][j]])
                cp.append(confidences[i][j])
    if pp:
        predictions_p.append(pp)
        confidences_p.append(cp)
    return predictions_p, confidences_p


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
        train_data = SLDataset(train_examples, training=True)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=config['train']['batch_size'])
        total_train_steps = math.ceil(len(train_data) / config['train']['batch_size']) * config['train']['epochs']

        dev_data = SLDataset(processor.get_examples(data_file=config['eval']['data_file'], set_type='dev'), training=False)

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
        dev_f1 = {}
        least_save_f1 = 0
        for _ in trange(0, config['train']['epochs'], desc='Epoch'):
            for step, batch in enumerate(tqdm(train_dataloader, desc='Iteration')):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, predict_mask, one_hot_labels = batch
                loss = model(input_ids, input_mask, predict_mask, one_hot_labels)
                summary.add_scalar('loss', loss, global_step)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                global_step += 1
                if global_step > config['eval']['begin'] and global_step % config['eval']['interval'] == 0:
                    model.eval()
                    predictions, _ = inference(model, dev_data, config['eval']['batch_size'], device)
                    model.train()
                    measure = SpanBasedF1Measure()
                    measure(predictions, dev_data.labels)
                    f1 = measure.get_metric()['f1-measure-overall']
                    summary.add_scalar('dev_f1', f1, global_step)
                    save_ckpt = False
                    remove_ckpt = None
                    if f1 > least_save_f1:
                        save_ckpt = True
                        dev_f1[global_step] = f1
                        if len(dev_f1) > config['eval']['max_save']:
                            remove_ckpt = sorted(dev_f1.items(), key=lambda x:x[1])[0][0]
                            least_save_f1 = dev_f1.pop(remove_ckpt)
                    if save_ckpt:
                        model_to_save = model.module if hasattr(model, 'module') else model
                        torch.save({'model_state': model_to_save.state_dict(), 'max_seq_length': max_seq_length,
                                    'label_list': label_list, 'lower_case': config['task']['lower_case']},
                                   os.path.join(config['task']['output_dir'], 'checkpoint-{}'.format(global_step)))
                    if remove_ckpt is not None:
                        os.system('rm {}'.format(os.path.join(config['task']['output_dir'], 'checkpoint-{}'.format(remove_ckpt))))
        dev_f1 = sorted(dev_f1.items(), key=lambda x:x[1], reverse=True)
        logger.info(dev_f1)
        with open(os.path.join(config['task']['output_dir'], 'dev_score.txt'), 'w', encoding='utf-8') as f:
            f.write(str(dev_f1)+'\n')
        best_ckpt = os.path.join(config['task']['output_dir'], 'checkpoint-{}'.format(dev_f1[0][0]))
        logging.info('Load %s' % best_ckpt)
        best_checkpoint = torch.load(best_ckpt, map_location='cpu')
        model = BertForSL.from_pretrained(config['task']['bert_model_dir'], state_dict=best_checkpoint['model_state'],
                                          num_labels=len(label_list), dropout_rate=config['train']['dropout_rate'])
        model.to(device)

    if config['predict']['do']:

        predict_data = SLDataset(processor.get_examples(data_file=config['predict']['data_file'], set_type='predict'), training=False)
        model.eval()
        predictions, confidences = inference(model, predict_data, config['predict']['batch_size'], device)
        if predict_data.labels is not None:
            measure = SpanBasedF1Measure()
            measure(predictions, predict_data.labels)
            logger.info(measure.get_metric())
            with open(os.path.join(config['task']['output_dir'], 'metrics.txt'), 'w', encoding='utf-8') as writer:
                writer.write(str(measure.get_metric()))
        with open(os.path.join(config['task']['output_dir'], 'predictions.txt'), 'w', encoding='utf-8') as writer:
            for item in predictions:
                writer.write(' '.join(item)+'\n')
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

    if not os.path.exists(os.path.join(config['task']['output_dir'], 'config.txt')):
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
    processor = processors[config['task']['task_name']]()
    label_list = processor.get_labels()
    label_map = {label: i for i, label in enumerate(label_list)}
    logger.info(label_map)
    tokenizer = BertTokenizer.from_pretrained(config['task']['bert_model_dir'],
                                              do_lower_case=config['task']['lower_case'])
    if config['task']['checkpoint']:
        logging.info('Load %s' % config['task']['checkpoint'])
        checkpoint = torch.load(config['task']['checkpoint'], map_location='cpu')
        model = BertForSL.from_pretrained(config['task']['bert_model_dir'], state_dict=checkpoint['model_state'],
                                          num_labels=len(label_list), dropout_rate=config['train']['dropout_rate'])
    else:
        logger.info('Load %s' % config['task']['bert_model_dir'])
        model = BertForSL.from_pretrained(config['task']['bert_model_dir'], num_labels=len(label_list),
                                          dropout_rate=config['train']['dropout_rate'])
    model.to(device)

    main()
