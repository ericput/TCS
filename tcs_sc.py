import os
import copy
import math
import logging
import yaml
import csv
import argparse
import pickle
import random
import jenkspy
from sklearn.metrics import accuracy_score
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
        model_config.attention_probs_dropout_prob = dropout_rate
        model_config.hidden_dropout_prob = dropout_rate
        super(BertForSC, self).__init__(model_config)
        self.bert = BertModel(model_config)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.hidden2label = torch.nn.Linear(model_config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, segment_ids, input_mask, label_id=None):
        repre, _ = self.bert(input_ids, segment_ids, attention_mask=input_mask, output_all_encoded_layers=False)
        repre = self.dropout(repre)
        repre = torch.mean(repre, dim=1)
        logits = self.hidden2label(repre)
        if label_id is not None:
            loss = F.cross_entropy(logits, label_id)
            return loss
        else:
            return F.softmax(logits, -1)


class Distillation(torch.nn.Module):

    def __init__(self, teacher, student, loss_func='mse'):
        super(Distillation, self).__init__()
        self.teacher = teacher
        self.student = student
        self.loss_func = loss_func

    def forward(self, input_ids, perturbed_input_ids, segment_ids, input_mask):
        teacher_p = self.teacher(input_ids.to(device1), segment_ids.to(device1), input_mask.to(device1)).to(device0)
        student_p = self.student(perturbed_input_ids, segment_ids, input_mask)
        if self.loss_func == 'mse':
            losses = torch.sum((teacher_p - student_p) ** 2, -1)
        else:
            losses = torch.sum(teacher_p*torch.log(teacher_p)-teacher_p*torch.log(student_p), -1)
        loss = torch.mean(losses)
        return loss


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
        for index, line in enumerate(open(data_file, 'r', encoding='utf-8')):
            guid = '%s-%d' % (set_type, index)
            label, content = line.strip().split('\t')
            examples.append(InputExample(guid=guid, text_a=eval(content).decode('utf-8'), text_b=None, label=label))
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


class ULTrainDataset(Dataset):
    def __init__(self, src_examples, tgt_examples, change_rate=0.0, **kwargs):
        self.change_rate = change_rate
        self.mask_id = tokenizer.vocab['[MASK]']
        self.kwargs = kwargs
        src_features = convert_examples_to_features(src_examples)
        self.src_tensors = [['src', torch.LongTensor(f.input_ids), torch.LongTensor(f.segment_ids), torch.BoolTensor(f.input_mask)] for f in src_features]
        tgt_example_shards = [[] for _ in range(kwargs['num_shard'])]
        for index, shard_id in enumerate(kwargs['shard_info']):
            tgt_example_shards[shard_id].append(tgt_examples[index])
        self.tgt_tensor_shards = []
        for shard_id in range(kwargs['num_shard']):
            features = convert_examples_to_features(tgt_example_shards[shard_id])
            self.tgt_tensor_shards.append([['tgt', torch.LongTensor(f.input_ids), torch.LongTensor(f.segment_ids), torch.BoolTensor(f.input_mask)] for f in features])
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
        length = sum(sample[3])
        indices = np.random.rand(length) < self.change_rate
        for index in range(length):
            if indices[index]:
                perturbed_input_ids[index] = self.mask_id
        return sample[1], perturbed_input_ids, sample[2], sample[3]


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
            input_ids, segment_ids, input_mask = batch
            batch_dists = infer_model(input_ids, segment_ids, input_mask)
            batch_confidences, batch_predictions = torch.max(batch_dists, -1)
            confidences.extend(batch_confidences.tolist())
            predictions.extend(batch_predictions.tolist())
    return predictions, confidences


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
        teacher_model = BertForSC.from_pretrained(config['task']['bert_model_dir'],
                                                  state_dict=teacher_checkpoint['model_state'],
                                                  dropout_rate=config['train']['dropout_rate'],
                                                  num_labels=len(label_map))
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
            pool_data = SCDataset(pool_examples, training=False)
            predictions, confidences = inference(teacher_model, pool_data, config['eval']['batch_size'], device1)
            for label_id in range(len(label_map)):
                subset_scores = [(confidences[i], i) for i in range(len(confidences)) if predictions[i] == label_id]
                subset_scores.sort(key=lambda x: x[0], reverse=True)
                for item in subset_scores[:config['train']['topk']]:
                    iter_info[pool_ids_map[item[1]]] = 1
        with open(os.path.join(config['task']['output_dir'], 'iter_info.bin'), 'wb') as handle:
            pickle.dump(iter_info, handle)
        iter_tgt_examples = [tgt_examples[i] for i in range(len(tgt_examples)) if iter_info[i] == 1]
        shard_info = [0] * len(iter_tgt_examples)
        tgt_data = SCDataset(iter_tgt_examples, training=False)
        predictions, confidences = inference(teacher_model, tgt_data, config['eval']['batch_size'], device1)
        for label_id in range(len(label_map)):
            subset = [confidences[i] for i in range(len(confidences)) if predictions[i] == label_id]
            subset_ids_map = [i for i in range(len(confidences)) if predictions[i] == label_id]
            subset_shard_info = sharding_function(subset, config['train']['num_shard'])
            for i in range(len(subset_shard_info)):
                shard_info[subset_ids_map[i]] = subset_shard_info[i]
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
        dev_data = SCDataset(dev_examples, training=False)

        logger.info('***** Running training *****')
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
        for epoch_id in trange(config['train']['num_epoch'], desc='Epoch'):
            train_data.regen(epoch_id)
            train_sampler = RandomSampler(train_data)
            train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=config['train']['batch_size'])
            for step, batch in enumerate(tqdm(train_dataloader, desc='Iteration')):
                batch = tuple(t.to(device0) for t in batch)
                input_ids, perturbed_input_ids, segment_ids, input_mask = batch
                loss = model(input_ids, perturbed_input_ids, segment_ids, input_mask)
                summary.add_scalar('teaching_loss', loss, global_step)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                global_step += 1
                if global_step > config['eval']['begin'] and global_step % config['eval']['interval'] == 0:
                    student_model.eval()
                    predictions, _ = inference(student_model, dev_data, config['eval']['batch_size'], device0)
                    student_model.train()
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
                        model_to_save = student_model.module if hasattr(student_model, 'module') else student_model
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
        student_model = BertForSC.from_pretrained(config['task']['bert_model_dir'], state_dict=best_checkpoint['model_state'],
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
        predict_data = SCDataset(predict_examples, training=False)
        student_model.eval()
        predictions, confidences = inference(student_model, predict_data, config['predict']['batch_size'], device0)
        if predict_data.label_ids is not None:
            score = metrics(predict_data.label_ids, predictions)
            logger.info(score)
            with open(os.path.join(config['task']['output_dir'], 'metrics.txt'), 'w', encoding='utf-8') as writer:
                writer.write(str(score))
        with open(os.path.join(config['task']['output_dir'], 'predictions.txt'), 'w', encoding='utf-8') as writer:
            for item in predictions:
                writer.write(str(item)+'\n')
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
    torch.cuda.manual_seed_all(config['train']['seed'])
    device0 = torch.device('cuda', 0)
    device1 = torch.device('cuda', 1)

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

    model.to(device0)

    main()
