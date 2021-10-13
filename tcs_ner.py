import os
import math
import pickle
import logging
import yaml
import copy
import random
import argparse
import jenkspy
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
        model_config.attention_probs_dropout_prob = dropout_rate
        model_config.hidden_dropout_prob = dropout_rate
        super(BertForSL, self).__init__(model_config)
        self.num_labels = num_labels
        self.bert = BertModel(model_config)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.hidden2label = torch.nn.Linear(
            model_config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, input_mask, predict_mask=None, one_hot_labels=None):
        repre, _ = self.bert(
            input_ids, attention_mask=input_mask, output_all_encoded_layers=False)
        repre = self.dropout(repre)
        logits = self.hidden2label(repre)
        p = softmax(logits, -1)
        if one_hot_labels is not None:
            losses = -torch.sum(one_hot_labels*torch.log(p), -1)
            loss = torch.sum(torch.masked_select(losses, predict_mask))
            return loss
        else:
            return p


class Distillation(torch.nn.Module):

    def __init__(self, teacher, student, loss_func='mse'):
        super(Distillation, self).__init__()
        self.teacher = teacher
        self.student = student
        self.loss_func = loss_func

    def forward(self, input_ids, perturbed_input_ids, input_mask, predict_mask):
        teacher_p = self.teacher(input_ids.to(
            device1), input_mask.to(device1)).to(device0)
        student_p = self.student(perturbed_input_ids, input_mask)
        if self.loss_func == 'mse':
            losses = torch.sum((teacher_p - student_p) ** 2, -1)
        else:
            losses = torch.sum(teacher_p*torch.log(teacher_p) -
                               teacher_p*torch.log(student_p), -1)
        loss = torch.sum(torch.masked_select(losses, predict_mask))
        return loss


class InputExample(object):

    def __init__(self, guid, units, labels=None):
        self.guid = guid
        self.units = units
        self.labels = labels


class InputFeatures(object):

    def __init__(self, input_ids, input_mask, predict_mask, one_hot_labels=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.predict_mask = predict_mask
        self.one_hot_labels = one_hot_labels


class DataProcessor(object):

    @staticmethod
    def create_examples_from_conll_format_file(data_file, set_type='data'):
        examples, units, labels = [], [], []
        for index, line in enumerate(open(data_file, encoding='utf-8')):
            segs = line.split()
            if not segs:
                guid = '%s-%d' % (set_type, index)
                examples.append(InputExample(
                    guid=guid, units=units, labels=labels))
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
            examples.append(InputExample(
                guid=guid, units=units, labels=labels))
        return examples

    @staticmethod
    def create_examples_from_files(content_file, label_file=None, set_type='data'):
        examples = []
        index = 0
        if label_file is not None:
            for content_line, label_line in zip(open(content_file, encoding='utf-8'), open(label_file, encoding='utf-8')):
                guid = '%s-%d' % (set_type, index)
                units = content_line.split()
                labels = label_line.split()
                assert len(units) == len(labels)
                for i in range(len(labels)):
                    if labels[i] not in label_map:
                        labels[i] = 'O'
                examples.append(InputExample(
                    guid=guid, units=units, labels=labels))
                index += 1
        else:
            for content_line in open(content_file, encoding='utf-8'):
                guid = '%s-%d' % (set_type, index)
                units = content_line.split()
                examples.append(InputExample(guid=guid, units=units))
                index += 1
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


def convert_examples_to_features(examples, use_label):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    line_tags = []
    for (ex_index, example) in enumerate(examples):

        if use_label:
            labels = example.labels
        else:
            labels = ['O'] * len(example.units)

        samples = []
        context, tokens, predict_mask, label_ids = [], [], [], []
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
                samples.append(
                    [['[CLS]'] + context + tokens[:l] + ['[SEP]'], [0] * (len(context) + 1) + predict_mask[:l] + [0],
                     [0] * (len(context) + 1) + label_ids[:l] + [0]])
                if not context:
                    line_tags.append(1)
                else:
                    line_tags.append(0)
                context = tokens[max(0, l - max_seq_length // 2):l]
                tokens, predict_mask, label_ids = tokens[l:
                    ], predict_mask[l:], label_ids[l:]
        if sum(predict_mask):
            samples.append([['[CLS]'] + context + tokens + ['[SEP]'], [0] * (len(
                context) + 1) + predict_mask + [0], [0] * (len(context) + 1) + label_ids + [0]])
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
            if use_label:
                label_ids = s[2] + zero_padding
                assert len(label_ids) == max_seq_length
                one_hot_labels = np.eye(
                    len(label_map), dtype=np.float32)[label_ids]
            else:
                one_hot_labels = None
            features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask,
                            predict_mask=predict_mask, one_hot_labels=one_hot_labels))
    assert len(examples) == sum(line_tags), logger.error(
        '{} != {}'.format(len(examples), sum(line_tags)))
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
            self.tensors = [[torch.LongTensor(f.input_ids), torch.LongTensor(
                f.input_mask)] for f in features]

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, item):
        return self.tensors[item]


class ULTrainDataset(Dataset):
    def __init__(self, src_examples, tgt_examples, change_rate=0.0, **kwargs):
        self.change_rate = change_rate
        self.mask_id = tokenizer.vocab['[MASK]']
        self.kwargs = kwargs
        src_features, _ = convert_examples_to_features(src_examples, False)
        self.src_tensors = [['src', torch.LongTensor(f.input_ids), torch.LongTensor(
            f.input_mask), torch.BoolTensor(f.predict_mask)] for f in src_features]
        tgt_example_shards = [[] for _ in range(kwargs['num_shard'])]
        for index, shard_id in enumerate(kwargs['shard_info']):
            tgt_example_shards[shard_id].append(tgt_examples[index])
        self.tgt_tensor_shards = []
        for shard_id in range(kwargs['num_shard']):
            features, _ = convert_examples_to_features(
                tgt_example_shards[shard_id], False)
            self.tgt_tensor_shards.append([['tgt', torch.LongTensor(f.input_ids), torch.LongTensor(
                f.input_mask), torch.BoolTensor(f.predict_mask)] for f in features])
        self.tensors = None

    def regen(self, epoch_id):
        tgt_tensors = []
        if self.kwargs['cl_strategy'] == 0:
            logger.info('Using strategy 0')
            for i in range(min(epoch_id + 1, self.kwargs['num_shard'])):
                tgt_tensors.extend(self.tgt_tensor_shards[i])
        elif self.kwargs['cl_strategy'] == 1:
            logger.info('Using strategy 1')
            for i in range(max(1, self.kwargs['num_shard']-epoch_id)):
                tgt_tensors.extend(self.tgt_tensor_shards[i])
        else:
            logger.info('Using strategy 2')
            for i in range(self.kwargs['num_shard']):
                tgt_tensors.extend(self.tgt_tensor_shards[i])
        self.tensors = self.src_tensors + tgt_tensors

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, item):
        sample = self.tensors[item]
        if sample[0] == 'src':
            return sample[1], sample[1], sample[2], sample[3]
        perturbed_input_ids = copy.deepcopy(sample[1])
        perturbed_predict_mask = copy.deepcopy(sample[3])
        length = sum(sample[2])
        indices = np.random.rand(length) < self.change_rate
        for index in range(length):
            if indices[index]:
                perturbed_input_ids[index] = self.mask_id
                perturbed_predict_mask[index] = 0
        return sample[1], perturbed_input_ids, sample[2], perturbed_predict_mask


def jenks_breaks(scores, num_shard):
    shard_info = [0] * len(scores)
    if num_shard == 1:
        return shard_info
    breaks = jenkspy.jenks_breaks(scores, nb_class=num_shard)
    breaks.pop(0)
    for i in range(len(scores)):
        for j in range(num_shard):
            if scores[i] <= breaks[j]:
                shard_info[i] = num_shard-1-j
                break
    return shard_info

def base_breaks(scores, num_shard):
    shard_info = [0] * len(scores)
    if num_shard == 1:
        return shard_info
    indexed_scores = [(i, scores[i]) for i in range(len(scores))]
    indexed_scores.sort(key=lambda x:x[-1], reverse=True)
    width = len(scores)//num_shard
    for i in range(num_shard):
        for item in indexed_scores[i*width:(i+1)*width]:
            shard_info[item[0]] = i
    return shard_info


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
    student_model = model

    if config['train']['do']:

        if config['task']['frozen_level'] > 0:
            for n, p in student_model.named_parameters():
                if 'embedding' in n:
                    p.requires_grad = False
                elif 'encoder' in n:
                    if config['task']['frozen_level'] > 1:
                        if int(n.split('.')[3]) < 3:
                            p.requires_grad = False

        teacher_checkpoint = torch.load(config['task']['teacher_checkpoint'], map_location='cpu')
        teacher_model = BertForSL.from_pretrained(config['task']['bert_model_dir'],
                                                  state_dict=teacher_checkpoint['model_state'],
                                                  dropout_rate=config['train']['dropout_rate'],
                                                  num_labels=len(label_list))
        teacher_model.to(device1)
        for p in teacher_model.parameters():
            p.requires_grad = False
        teacher_model.eval()
        model = Distillation(teacher=teacher_model, student=student_model, loss_func=config['train'].get('loss_func', 'mse'))

        if 'sharding_function' in config['train'] and config['train']['sharding_function'] == 'base':
            logger.info('base breaks')
            sharding_function = base_breaks
        else:
            logger.info('jenks breaks')
            sharding_function = jenks_breaks
        
        src_data_file = config['train']['data_file'][0]
        src_examples = processor.get_examples(data_file=src_data_file, set_type='train')
        logger.info('src data: %s' % src_data_file)
        tgt_examples = []
        for tgt_data_file in config['train']['data_file'][1:]:
            logger.info('tgt data: %s' % tgt_data_file)
            tgt_examples.extend(processor.get_examples(data_file=tgt_data_file, set_type='train'))
        if config['train']['iter_info']:
            iter_info = pickle.load(open(config['train']['iter_info'], 'rb'))
        else:
            iter_info = [0] * len(tgt_examples)
        pool_examples = [tgt_examples[i] for i in range(len(tgt_examples)) if iter_info[i] == 0]
        pool_ids_map = [i for i in range(len(tgt_examples)) if iter_info[i] == 0]
        if pool_examples:
            pool_data = SLDataset(pool_examples, training=False)
            predictions, confidences = inference(teacher_model, pool_data, config['eval']['batch_size'], device1)
            scores = [(sum(list(map(np.log, cline))) / len(cline), i) for i, cline in enumerate(confidences)]
            scores.sort(key=lambda x:x[0], reverse=True)
            for item in scores[:config['train']['topk']]:
                iter_info[pool_ids_map[item[1]]] = 1
        with open(os.path.join(config['task']['output_dir'], 'iter_info.bin'), 'wb') as handle:
            pickle.dump(iter_info, handle)
        iter_tgt_examples = [tgt_examples[i] for i in range(len(tgt_examples)) if iter_info[i] == 1]
        tgt_data = SLDataset(iter_tgt_examples, training=False)
        predictions, confidences = inference(teacher_model, tgt_data, config['eval']['batch_size'], device1)
        scores = [sum(list(map(np.log, cline)))/len(cline) for cline in confidences]
        shard_info = sharding_function(scores, config['train']['num_shard'])
        with open(os.path.join(config['task']['output_dir'], 'shard_info.bin'), 'wb') as handle:
            pickle.dump(shard_info, handle)
        ds_args = {}
        ds_args['num_shard'] = config['train']['num_shard']
        ds_args['shard_info'] = shard_info
        ds_args['cl_strategy'] = config['train'].get('cl_strategy', 0)
        train_data = ULTrainDataset(src_examples, iter_tgt_examples, **ds_args)
        total_train_steps = 0
        for epoch_id in range(config['train']['num_epoch']):
            train_data.regen(epoch_id)
            total_train_steps += len(train_data)
        total_train_steps = math.ceil(total_train_steps / config['train']['batch_size'])

        dev_examples = []
        if isinstance(config['eval']['data_file'], str):
            dev_files = [config['eval']['data_file']]
        else:
            dev_files = config['eval']['data_file']
        for dev_data_file in dev_files:
            logger.info('dev data: %s' % dev_data_file)
            dev_examples.extend(processor.get_examples(data_file=dev_data_file, set_type='dev'))
        dev_data = SLDataset(dev_examples, training=False)

        logger.info('***** Running training *****')
        logger.info('  Batch size = %d', config['train']['batch_size'])
        logger.info('  Num steps = %d', total_train_steps)

        all_params = [(n, p) for n, p in student_model.named_parameters() if p.requires_grad]
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
        for epoch_id in trange(config['train']['num_epoch'], desc='Epoch'):
            train_data.regen(epoch_id)
            train_sampler = RandomSampler(train_data)
            train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=config['train']['batch_size'])
            for batch in tqdm(train_dataloader, desc='Iteration'):
                batch = tuple(t.to(device0) for t in batch)
                input_ids, perturbed_input_ids, input_mask, predict_mask = batch
                loss = model(input_ids, perturbed_input_ids, input_mask, predict_mask)
                summary.add_scalar('teaching_loss', loss, global_step)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                global_step += 1
                if global_step > config['eval']['begin'] and global_step % config['eval']['interval'] == 0:
                    student_model.eval()
                    predictions, _ = inference(student_model, dev_data, config['eval']['batch_size'], device0)
                    student_model.train()
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
                        model_to_save = student_model.module if hasattr(student_model, 'module') else student_model
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
        student_model = BertForSL.from_pretrained(config['task']['bert_model_dir'], state_dict=best_checkpoint['model_state'],
                                                  num_labels=len(label_map), dropout_rate=config['train']['dropout_rate'])
        student_model.to(device0)

    if config['predict']['do']:

        predict_examples = []
        if isinstance(config['predict']['data_file'], str):
            predict_files = [config['predict']['data_file']]
        else:
            predict_files = config['predict']['data_file']
        for predict_data_file in predict_files:
            logger.info('predict data: %s' % predict_data_file)
            predict_examples.extend(processor.get_examples(data_file=predict_data_file, set_type='predict'))
        predict_data = SLDataset(predict_examples, training=False)
        student_model.eval()
        predictions, confidences = inference(student_model, predict_data, config['predict']['batch_size'], device0)
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
    torch.cuda.manual_seed_all(config['train']['seed'])
    device0 = torch.device('cuda', 0)
    device1 = torch.device('cuda', 1)

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
    model.to(device0)

    main()
