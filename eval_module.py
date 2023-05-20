import torch
import torch.nn as nn
import os
import json
import tabulate
import random
import time
from utils import Acc_Per_Context, Acc_Per_Context_Class, cal_acc, save_model, load_model
import numpy as np

@torch.no_grad()
def eval_training(config, args, net, test_loader, loss_function, writer, epoch=0, tb=True):
    start = time.time()
    if isinstance(net, list):
        for net_ in net:
            net_.eval()
    else:
        net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0
    acc_per_context = Acc_Per_Context(config['cxt_dic_path'])
    rst=[[0,0],[0,0]]

    for (images, labels, context) in test_loader:

        images = images.cuda()
        labels = labels.cuda()

        if isinstance(net, list):
            feature = net[-1](images)
            if isinstance(feature, list):
                feature = feature[0]  # chosse causal feature
            try:
                W_mean = torch.stack([net_.fc.weight for net_ in net[:config['variance_opt']['n_env']]], 0).mean(0)
            except:
                W_mean = torch.stack([net_.module.fc.weight for net_ in net[:config['variance_opt']['n_env']]], 0).mean(0)
            outputs = nn.functional.linear(feature, W_mean)
        else:
            outputs = net(images)
        loss = loss_function(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        for i in range(labels.shape[0]):
            rst[labels[i]][preds[i]]+=1
        correct += preds.eq(labels).sum()
        # acc_per_context.update(preds, labels, context)

    finish = time.time()
    print('Evaluating Network.....')
    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        test_loss / len(test_loader.dataset),
        correct.float() / len(test_loader.dataset),
        finish - start
    ))
    # print('Evaluate Acc Per Context...')
    # acc_cxt = acc_per_context.cal_acc()
    # print(tabulate.tabulate(acc_cxt, headers=['Context', 'Acc'], tablefmt='grid'))

    #add informations to tensorboard
    if tb:
        writer.add_scalar('Test/Average loss', test_loss / len(test_loader.dataset), epoch)
        writer.add_scalar('Test/Accuracy', correct.float() / len(test_loader.dataset), epoch)

    print('[rst]:')
    print(rst[0])
    print(rst[1])
    TPR=rst[1][1]/(rst[1][0]+rst[1][1])*100
    TNR=rst[0][0]/(rst[0][0]+rst[0][1])*100
    print(TPR,'%')
    print(TNR,'%')
    print(TPR*TNR/100,'%')

    return correct.float() / len(test_loader.dataset)


@torch.no_grad()
def eval_best(config, args, net, test_loader, loss_function ,checkpoint_path, best_epoch):
    start = time.time()
    try:
        load_model(net, checkpoint_path.format(net=args.net, epoch=best_epoch, type='best'))
        # net.load_state_dict(torch.load(checkpoint_path.format(net=args.net, epoch=best_epoch, type='best')))
    except:
        print('no best checkpoint')
        load_model(net, checkpoint_path.format(net=args.net, epoch=180, type='regular'))
        # net.load_state_dict(torch.load(checkpoint_path.format(net=args.net, epoch=180, type='regular')))
    if isinstance(net, list):
        for net_ in net:
            net_.eval()
    else:
        net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0
    label2train = test_loader.dataset.label2train
    label2train = {v: k for k, v in label2train.items()}
    acc_per_context = Acc_Per_Context_Class(config['cxt_dic_path'], list(label2train.keys()))

    for (images, labels, context) in test_loader:

        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        if isinstance(net, list):
            feature = net[-1](images)
            if isinstance(feature, list):
                feature = feature[0]  # chosse causal feature
            try:
                W_mean = torch.stack([net_.fc.weight for net_ in net[:config['variance_opt']['n_env']]], 0).mean(0)
            except:
                W_mean = torch.stack([net_.module.fc.weight for net_ in net[:config['variance_opt']['n_env']]], 0).mean(0)
            outputs = nn.functional.linear(feature, W_mean)
        else:
            outputs = net(images)
        loss = loss_function(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()
        acc_per_context.update(preds, labels, context)

    finish = time.time()

    print('Evaluating Network.....')
    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        test_loss / len(test_loader.dataset),
        correct.float() / len(test_loader.dataset),
        finish - start
    ))
    print('Evaluate Acc Per Context Per Class...')
    class_dic = json.load(open(config['class_dic_path'], 'r'))
    class_dic = {v: k for k, v in class_dic.items()}
    acc_cxt_all_class = acc_per_context.cal_acc()
    for label_class in acc_cxt_all_class.keys():
        acc_class = acc_cxt_all_class[label_class]
        print('Class: %s' %(class_dic[int(label2train[label_class])]))
        print(tabulate.tabulate(acc_class, headers=['Context', 'Acc'], tablefmt='grid'))

    return correct.float() / len(test_loader.dataset)

from torchcam.methods import SmoothGradCAMpp

@torch.no_grad()
def eval_mode(config, args, net, test_loader, loss_function, model_path,mode='normal'):
    start = time.time()
    load_model(net, model_path)
    if isinstance(net, list):
        for net_ in net:
            net_.eval()
    else:
        net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0
    label2train = test_loader.dataset.label2train
    label2train = {v: k for k, v in label2train.items()}
    acc_per_context = Acc_Per_Context_Class(config['cxt_dic_path'], list(label2train.keys()))

    roc=np.zeros((3,len(test_loader.dataset)))
    roc_idx=0

    for (images, labels, context) in test_loader:

        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        if isinstance(net, list):
            feature = net[-1](images)
            feature2=feature[-1][0].to('cpu').numpy()
            if isinstance(feature, list):
                feature = feature[0]  # chosse causal feature
            try:
                W_mean = torch.stack([net_.fc.weight for net_ in net[:config['variance_opt']['n_env']]], 0).mean(0)
            except:
                W_mean = torch.stack([net_.module.fc.weight for net_ in net[:config['variance_opt']['n_env']]], 0).mean(0)
            outputs = nn.functional.linear(feature, W_mean)

            if mode=='act_map':
                print('draw act map')
                cam_extractor=SmoothGradCAMpp(net)
                activation_map = cam_extractor(outputs.squeeze(0).argmax().item(), outputs)
                
            rst=outputs.to('cpu').numpy().argmax(1)
            roc[0][roc_idx:roc_idx+len(labels)]=labels.cpu().numpy()
            # roc[1][roc_idx:roc_idx+len(labels)]=torch.nn.Softmax(dim=1)(outputs)[:,1].cpu().numpy()
            # roc[1][roc_idx:roc_idx+len(labels)]=torch.nn.Softmax(dim=1)(outputs)[:,0].cpu().numpy()
            # roc[2][roc_idx:roc_idx+len(labels)]=torch.nn.Softmax(dim=1)(outputs)[:,1].cpu().numpy()
            roc[1][roc_idx:roc_idx+len(labels)]=outputs[:,0].cpu().numpy()
            roc[2][roc_idx:roc_idx+len(labels)]=outputs[:,1].cpu().numpy()
            # roc[1][roc_idx:roc_idx+len(labels)]=torch.nn.Softmax(dim=1)(outputs).gather(1,outputs.argmax(1).unsqueeze(1)).squeeze(1).cpu().numpy()
            roc_idx+=len(labels)

            # for i in range(feature2.shape[0]):
            #     fn=os.path.join('/home/zxz/proj/caam/0-NICO/feature',str(rst[i])+'_'+context[i].removesuffix('.jpg')+'.npy')
            #     np.save(open(fn,'wb'),feature2[i])
                
        else:
            outputs = net(images)

        np.save('roc.npy',roc)

        loss = loss_function(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()
        # acc_per_context.update(preds, labels, context)

    finish = time.time()

    print('Evaluating Network.....')
    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        test_loss / len(test_loader.dataset),
        correct.float() / len(test_loader.dataset),
        finish - start
    ))

    return correct.float() / len(test_loader.dataset)