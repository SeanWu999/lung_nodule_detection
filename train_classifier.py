from config import opt
from net.resnet import ResNet18
import torch as t
from torch.nn import init
from dataloader.dataloader import clsDataLoader, clsValDataLoader
import torchnet as tnt
import warnings
import time
warnings.filterwarnings("ignore")

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv3d') != -1:
        init.xavier_normal_(m.weight.data)
        init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data)
        init.constant_(m.bias.data, 0)

def train():
    loss_function = t.nn.CrossEntropyLoss()
    model = ResNet18().cuda()

    if opt.cls_model_path is not None:
        model.load_state_dict(t.load(opt.cls_model_path))
    else:
        model.apply(weights_init)

    dataset = clsDataLoader()
    dataloader = t.utils.data.DataLoader(dataset, opt.batch_size,
                                         num_workers=2,
                                         shuffle=True,
                                         pin_memory=opt.pin_memory)

    val_dataset = clsValDataLoader()
    val_dataloader = t.utils.data.DataLoader(val_dataset, opt.batch_size,
                                             num_workers=2,
                                             shuffle=False,
                                             pin_memory=opt.pin_memory)

    lr = opt.learningRate

    optimizer = t.optim.Adam(model.parameters(), lr=lr, weight_decay=opt.weight_decay)
    loss_meter = tnt.meter.AverageValueMeter()
    val_loss_meter = tnt.meter.AverageValueMeter()

    for epoch in range(opt.epochs):
        model.train()
        loss_meter.reset()

        for i, (input, label) in enumerate(dataloader):
            optimizer.zero_grad()
            input = t.autograd.Variable(input).cuda()
            target = label.cuda()

            output = model(input)
            loss = loss_function(output, target)
            loss_meter.add(loss.cpu().detach().numpy())

            loss.backward()
            optimizer.step()

        print("epoch:%4d, learning rate: %.8f, loss value: %.8f" % (epoch, lr, loss_meter.value()[0]))

        if epoch % opt.test_interval == 0 and epoch > 0:

            model.eval()
            tn = 0
            tp = 0
            fp = 0
            fn = 0
            testacc = 0

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
                testacc += int(num_correct.cpu().detach().numpy())

                val_loss = loss_function(val_output, val_target)
                val_loss_meter.add(val_loss.cpu().detach().numpy())

            testacc = testacc / opt.num_testDatasets
            print("----------------eval------------------")
            print("validating:loss value: %.8f" % val_loss_meter.value()[0])
            print("validating:accucacy: ", testacc)
            print("tp:", tp, "tn:", tn, "fp:", fp, "fn:", fn)
            print("recall:", tp / (tp + fn + opt.eps))
            print("precision:", tp / (tp + fp + opt.eps))
            print("----------------eval------------------")
            t.save(model.state_dict(), opt.cls_model_save + str(epoch)+'.pkl')

        if epoch % opt.decay_interval == 0 and epoch > 0:
            lr = lr * opt.lr_decay
            adjust_learning_rate(optimizer, lr)

if __name__ == "__main__":
    train()
