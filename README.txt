pytorch_pretrained_bert: 实现了BERT模型类及其优化器类和Tokenization类
tcs_ner.py: 实现了用于命名实体识别任务的TCS学习框架
tcs_sc.py：实现了用于文本分类任务的TCS学习框架
example_configs：配置文件示例
train.py：配合tcs_ner.py或tcs_sc.py进行迭代式训练，比如python train.py tcs_ner.py example_configs/ner_tgt_de.yaml 10
ner.py：命名实体识别，配合example_configs/ner_src_en.yaml训练源模型
sc.py：文本分类，配合example_configs/tc_src_en.yaml训练源模型