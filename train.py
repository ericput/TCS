import os
import sys
import yaml
import pickle
config = yaml.load(open(sys.argv[2]).read())
config_bin_path = sys.argv[2][:-4]+'bin'
output_dir = config['task']['output_dir']
num_iters = int(sys.argv[3])
if len(sys.argv) > 4:
    begin_iter = int(sys.argv[4])
else:
    begin_iter = 1
for i in range(begin_iter, num_iters+1):
    config['task']['output_dir'] = output_dir+'/%d'%i
    if i > 1:
        last_od = output_dir+'/%d'%(i-1)
        dev_score_path = last_od + '/' + 'dev_score.txt'
        dev_score = eval(open(dev_score_path).read().strip())
        ckpt = 'checkpoint-%d' % dev_score[0][0]
        config['task']['teacher_checkpoint'] = last_od + '/' + ckpt
        if 'iter_info' in config['train']:
            config['train']['iter_info'] = last_od+'/iter_info.bin'
    with open(config_bin_path, 'wb') as handle:
        pickle.dump(config, handle)
    os.system('python {} --config_bin {}'.format(sys.argv[1], config_bin_path))
