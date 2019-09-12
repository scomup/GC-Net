#!/usr/bin/env python3

import torch
import torchvision
import torchvision.transforms as transforms
import sys
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
import cv2
from torch.utils.data import Dataset, DataLoader

from torch.autograd import Variable
import numpy as np
from read_data import sceneDisp
import torch.optim as optim

from gc_net import *
from python_pfm import *

import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
torch.backends.cudnn.benchmark = True

#preprocess
def normalizeRGB(img):
    # for i in range(3):
    #     minval=torch.min(img[:,i, :, :])
    #     maxval=torch.max(img[:,i, :, :])
    #     if (minval.data!=maxval.data).cpu().numpy():
    #         img[:, i, :, :]=torch.div(img[:,i, :, :]-minval,maxval-minval)
    #         img[:,i,:,:]=torch.div(img[:,i, :, :]-0.5,0.5)
    # return img
    return img

def print_gpu_info():
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3, 1), 'GB')


mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)



tsfm=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean.tolist(), std.tolist())])
unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())




#h=256
#w=512

h=128
w=256


maxdisp=96 #gc_net.py also need to change  must be a multiple of 32...maybe can cancel the outpadding of deconv
batch=2
net = GcNet(h,w,maxdisp)
#net=net.cuda()
net=torch.nn.DataParallel(net).cuda()

show = False
show = True
#print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in net.parameters()])))

#train
def train(epoch_total,loadstate):

    loss_mul_list = []
    for d in range(maxdisp):
        loss_mul_temp = Variable(torch.Tensor(np.ones([batch, 1, h, w]) * d)).cuda()
        loss_mul_list.append(loss_mul_temp)
    loss_mul = torch.cat(loss_mul_list, 1)

    optimizer = optim.RMSprop(net.parameters(), lr=0.001, alpha=0.9)
    dataset = sceneDisp('','train',tsfm)
    _,H,W = dataset.__getitem__(0)['imL'].shape
    loss_fn=nn.L1Loss()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=True,num_workers=1)

    imL = Variable(torch.FloatTensor(1).cuda())
    imR = Variable(torch.FloatTensor(1).cuda())
    dispL = Variable(torch.FloatTensor(1).cuda())

    loss_list=[]
    start_epoch=0

    writer = SummaryWriter()
    n_iter = 0

    if loadstate==True:
        checkpoint = torch.load('./checkpoint/ckpt.t7')
        net.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']
        accu=checkpoint['accur']

    #print('startepoch:%d accuracy:%f' %(start_epoch,accu))
    
    for epoch in range(start_epoch,epoch_total):
        net.train()
        data_iter = iter(dataloader)

        #print('\nEpoch: %d' % epoch)
        train_loss=0
        acc_total=0
        for step in range(len(dataloader)-1):
            #print('----epoch:%d------step:%d------' %(epoch,step))
            data = next(data_iter)

            randomH = np.random.randint(0, H-h-1)
            randomW = np.random.randint(0, W-w-1)
            imageL = data['imL'][:,:,randomH:(randomH+h),randomW:(randomW+w)]
            imageR = data['imR'][:, :, randomH:(randomH + h), randomW:(randomW + w)]
            disL = data['dispL'][:, :, randomH:(randomH + h), randomW:(randomW + w)]
            imL.resize_(imageL.size()).copy_(imageL)
            imR.resize_(imageR.size()).copy_(imageR)
            dispL.resize_(disL.size()).copy_(disL)
            #normalize
            # imgL=normalizeRGB(imL)
            # imgR=normalizeRGB(imR)

            net.zero_grad()
            optimizer.zero_grad()

            x = net(imL,imR)
            # print(x.shape)
            # print(loss_mul.shape)
            # print(net)
            result=torch.sum(x.mul(loss_mul),1)
            result = result[:,None,:]
            # print(result.shape)
            #print_gpu_info()
            tt=loss_fn(result,dispL)
            #print_gpu_info()
            train_loss+=tt.item()
            # tt = loss(x, loss_mul, dispL)
            tt.backward()
            optimizer.step()


            result=result.view(batch,1,h,w)
            diff=torch.abs(result.cpu()-dispL.cpu())
            accuracy=torch.sum(diff<3)/float(h*w*batch)
            acc_total+=accuracy

            if(show):
                imL_ = unnormalize(imL[0]).permute(1,2,0).cpu().detach().numpy()
                imR_ = unnormalize(imR[0]).permute(1,2,0).cpu().detach().numpy()
                disp_TRUE_ = disL.cpu().detach().numpy()[0][0]
                disp_NET_ = result.cpu().detach().numpy()[0][0]
                plt.figure(figsize=(16, 8))
                plt.subplot(2,2,1)
                plt.imshow( imL_[...,::-1])
                plt.subplot(2,2,2)
                plt.imshow(imR_[...,::-1])
                plt.subplot(2,2,3)
                plt.imshow(disp_TRUE_,cmap='rainbow',vmin=0, vmax=maxdisp)
                plt.colorbar()
                plt.subplot(2,2,4)
                plt.imshow(disp_NET_,cmap='rainbow',vmin=0, vmax=maxdisp)
                plt.colorbar()

                plt.show()

            show_n = 10
            if step % show_n == (show_n - 1):
                writer.add_scalar('Loss/train_loss', train_loss/show_n, n_iter)
                #imL_ = unnormalize(imL[0])
                #disp_NET_ = result[0]
                #writer.add_image('Image/left', imL_, n_iter)
                #writer.add_image('Image/disparity', disp_NET_, n_iter)
                writer.close()


                n_iter += 1
                print('[%d, %5d, %5d] train_loss %.5f' % (
                    epoch + 1, step + 1, len(dataloader), train_loss/show_n))
                train_loss = 0.0


            #print('====accuracy for the result less than 3 pixels===:%f' %accuracy)
            #print('====average accuracy for the result less than 3 pixels===:%f' % (acc_total/(step+1)))

            # save
            if step%1000==0:
                state={'net':net.state_dict(),'step':step,
                       'loss_list':loss_list,'epoch':epoch,'accur':acc_total}
                torch.save(state,'checkpoint/ckpt.t7')
    fp.close()


#test
def test(loadstate):

    if loadstate==True:
        checkpoint = torch.load('./checkpoint/ckpt.t7')
        net.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']
        accu=checkpoint['accur']
    net.eval()
    imL = Variable(torch.FloatTensor(1).cuda())
    imR = Variable(torch.FloatTensor(1).cuda())
    dispL = Variable(torch.FloatTensor(1).cuda())

    dataset = sceneDisp('', 'test',tsfm)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    data_iter = iter(dataloader)
    data = next(data_iter)

    randomH = np.random.randint(0, 160)
    randomW = np.random.randint(0, 400)
    print('test')
    imageL = data['imL'][:, :, randomH:(randomH + h), randomW:(randomW + w)]
    imageR = data['imR'][:, :, randomH:(randomH + h), randomW:(randomW + w)]
    disL = data['dispL'][:, :, randomH:(randomH + h), randomW:(randomW + w)]
    imL.resize_(imageL.size()).copy_(imageL)
    imR.resize_(imageR.size()).copy_(imageR)
    dispL.resize_(disL.size()).copy_(disL)
    loss_mul_list_test = []
    for d in range(maxdisp):
        loss_mul_temp = Variable(torch.Tensor(np.ones([1, 1, h, w]) * d)).cuda()
        loss_mul_list_test.append(loss_mul_temp)
    loss_mul_test = torch.cat(loss_mul_list_test, 1)

    with torch.no_grad():
        result=net(imL,imR)

    disp=torch.sum(result.mul(loss_mul_test),1)
    diff = torch.abs(disp.cpu() -dispL.cpu())  # end-point-error

    accuracy = torch.sum(diff < 3) / float(h * w)
    print('test accuracy less than 3 pixels:%f' %accuracy)

    # save
    im=disp.cpu().numpy().astype('uint8')
    im=np.transpose(im,(1,2,0))
    cv2.imwrite('test_result.png',im,[int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    gt=np.transpose(dispL[0,:,:,:].cpu().numpy(),(1,2,0))
    cv2.imwrite('test_gt.png',gt,[int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    return disp

def main():
    epoch_total=20
    load_state=True
    train(epoch_total,load_state)
    test(load_state)


if __name__=='__main__':
    import torch.multiprocessing as mp
    mp.set_start_method('spawn')
    main()
