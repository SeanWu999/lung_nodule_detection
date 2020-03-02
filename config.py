#encoding:utf-8

class DefaultConfig:
    learningRate = 0.001                           #学习率
    epochs = 100                                   #训练回合
    test_interval = 5                              #测试间隔回合数
    seg_model_save = './save/'                     #分割模型储存地址
    cls_model_save = './save/'                     #分类模型储存地址
    lr_decay = 0.5                                 #学习率衰减系数
    weight_decay = 1e-5                            #正则权重衰减
    decay_interval = 50                            #学习率衰减间隔
    pin_memory = True                              #数据从CPU->pin_memory—>GPU加速
    num_testDatasets = 1000                        #测试集样本总数，用于计算精确度和召回率
    cls_test_path = './data/cls/test'             #分类样本测试数据地址
    cls_train_path = './data/cls/train'           #分类样本训练数据地址
    seg_test_path = './data/seg/test'             #分割样本测试数据地址
    seg_train_path = './data/seg/train'           #分割样本训练数据地址
    batch_size = 1
    seg_model_path = None
    cls_model_path = None
    eps = 1e-5


class Config(DefaultConfig):
    '''
    在这里修改,覆盖默认值
    '''

opt = Config()