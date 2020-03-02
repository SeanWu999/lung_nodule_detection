from utils.loss import SoftDiceLoss
from net.SegRes import Segmentation
from config import opt
from dataloader.dataloader import SegDataLoader, SegvalDataLoader
import torch as t
import torchnet as tnt
from torch.nn import init
import time

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def weights_init(m):
    classname = m.__class__.__name__

    if classname.find('Conv3d') != -1:
        init.xavier_normal_(m.weight.data)
        init.constant_(m.bias.data, 0)

    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data)
        init.constant_(m.bias.data, 0)

if __name__ == "__main__":

    loss_function = SoftDiceLoss()
    model = Segmentation()
    model.cuda()

    if opt.seg_model_path is not None:
        model.load_state_dict(t.load(opt.seg_model_path))
    else:
        model.apply(weights_init)

    dataset = SegDataLoader()
    dataloader = t.utils.data.DataLoader(dataset, opt.batch_size,
                                         num_workers=2,
                                         shuffle=True,
                                         pin_memory=opt.pin_memory)

    val_dataset = SegvalDataLoader()
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

        for i, (input, mask) in enumerate(dataloader):
            optimizer.zero_grad()
            input = t.autograd.Variable(input, requires_grad=True).cuda()
            target = t.autograd.Variable(mask, requires_grad=True).cuda()
            output = model(input)

            loss = loss_function(output, target)
            loss_meter.add(loss.cpu().detach().numpy())
            loss.backward()
            optimizer.step()

        print("epoch:%4d, learning rate: %.8f, loss value: %.8f" % (epoch, lr, loss_meter.value()[0]))

        if epoch % opt.test_interval == 0 and epoch > 0:
            model.eval()
            val_loss_meter.reset()
            for j, (val_input, val_mask) in enumerate(val_dataloader):
                val_input = t.autograd.Variable(val_input, requires_grad=True).cuda()
                val_target = t.autograd.Variable(val_mask, requires_grad=True).cuda()

                val_output = model(val_input)
                val_loss = loss_function(val_output, val_target)
                val_loss_meter.add(val_loss.cpu().detach().numpy())

            print("----------------eval------------------")
            print("validating:loss value: %.8f" % val_loss_meter.value()[0])
            print("----------------eval------------------")

            t.save(model.state_dict(), opt.seg_model_save + str(epoch)+'.pkl')

        if epoch%opt.decay_interval == 0 and epoch > 0:
            lr = lr * opt.lr_decay
            adjust_learning_rate(optimizer, lr)




