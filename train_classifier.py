from net import loss as Loss_
from config import opt
from net.resnet import ResNet18,ResNetPlus,Simple,Simple2,ResNet19,ResNet32
from net.inception import Inception
import torch as t
from torch.nn import init
from dataloader.dataloader import clsDataLoader, clsValDataLoader
from utils.util import get_optimizer
import torchnet as tnt
import warnings
import time
warnings.filterwarnings("ignore")

def weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv3d') != -1:
        init.xavier_normal_(m.weight.data)
        init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data)
        init.constant_(m.bias.data, 0)

def train(savePath, weightDecay = 1e-5, epochs = 1000, learningRate = 0.001, eps=1e-6):
    loss_function = getattr(Loss_, opt.cls_loss_function)
    model = ResNet19().cuda()

    if opt.cls_model_path is not None:
        model.load_state_dict(t.load(opt.cls_model_path))
    else:
        # print("nothing to do")
        model.apply(weights_init)

    dataset = clsDataLoader()
    dataloader = t.utils.data.DataLoader(dataset, 48,
                                         num_workers=2,
                                         shuffle=opt.shuffle,
                                         pin_memory=opt.pin_memory)

    val_dataset = clsValDataLoader()
    val_dataloader = t.utils.data.DataLoader(val_dataset, 48,
                                             num_workers=2,
                                             shuffle=False,
                                             pin_memory=opt.pin_memory)

    lr = learningRate

    optimizer = get_optimizer(model, lr, weight_decay=weightDecay)
    loss_meter = tnt.meter.AverageValueMeter()
    val_loss_meter = tnt.meter.AverageValueMeter()

    # loss_fn = t.nn.L1Loss(reduce=False, size_average=False)

    # start=time.time()
    for epoch in range(epochs):
        model.train()
        loss_meter.reset()

        for i, (input, label) in enumerate(dataloader):
            optimizer.zero_grad()
            input = t.autograd.Variable(input).cuda()
            target = label.cuda()

            output = model(input)
            loss = loss_function(output, target)
            loss_meter.add(loss.data[0])

            loss.backward()
            optimizer.step()

        if epoch % 1 == 0:
            print("epoch:%4d, learning rate: %.8f, loss value: %.8f" % (epoch, lr, loss_meter.value()[0]))

        if epoch % 3 == 0:
            model.eval()

            tn = 0
            tp = 0
            fp = 0
            fn = 0

            val_loss_meter.reset()
            for i, (input, label) in enumerate(val_dataloader):

                # 输入输出转成tensor格式,并制定GPU训练
                val_input = t.autograd.Variable(input).cuda()
                val_target = label.cuda()
                val_output = model(val_input)
                _, predict = t.max(val_output, 1)
                for i in range(len(val_target)):
                    if predict[i] == val_target[i] and val_target[i] == 0:
                        tn += 1
                    elif predict[i] == val_target[i] and val_target[i] == 1:
                        tp += 1
                    elif predict[i] != val_target[i] and val_target[i] == 0:
                        fp += 1
                    elif predict[i] != val_target[i] and val_target[i] == 1:
                        fn += 1

                num_correct = (predict == val_target).sum()
                testacc += int(num_correct.data[0])

                val_loss = loss_function(val_output, val_target)
                val_loss_meter.add(val_loss.data[0])

            testacc = testacc / 747
            print("----------------eval------------------")
            print("validating:loss value: %.8f" % val_loss_meter.value()[0])
            print("validating:accucacy: ", testacc)
            print("tp:", tp, "tn:", tn, "fp:", fp, "fn:", fn)
            print("recall:", tp / (tp + fn + eps))
            print("precision:", tp / (tp + fp + eps))
            print("----------------eval------------------")
            t.save(model.state_dict(), './save/'+savePath+'/cls_model_1229_' + str(epoch) + '.pkl')

        if epoch % 100 == 0 and epoch > 0:
            lr = lr * 0.9
            optimizer = get_optimizer(model, lr)

if __name__ == "__main__":
    train('1229_resnet', weightDecay=5e-6, epochs=2000, learningRate=0.01)
    #time.sleep(60)
    #print("####################################################################################")
    #train('1227_resnet2', weightDecay=2e-5, epochs=500, learningRate = 0.002)
    #train(epochs = 1000, savePath = '1227_resnet')