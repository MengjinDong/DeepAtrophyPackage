import torch
from torch import nn
from . import longi_models


def generate_model(opt):
    assert opt.model in [
        'resnet'
    ]

    if opt.model == 'resnet':
        assert opt.model_depth in [10, 18, 34, 50, 101, 152, 200]
        
        if opt.model_depth == 10:
            model = longi_models.resnet10(
                in_channel = opt.input_channels,
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_date_diff_classes=opt.num_date_diff_classes,
                )
        elif opt.model_depth == 18:
            model = longi_models.resnet18(
                in_channel=opt.input_channels,
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_date_diff_classes=opt.num_date_diff_classes,)
        elif opt.model_depth == 34:
            model = longi_models.resnet34(
                in_channel=opt.input_channels,
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_date_diff_classes=opt.num_date_diff_classes,)
        elif opt.model_depth == 50:
            model = longi_models.resnet50(
                in_channel=opt.input_channels,
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_date_diff_classes=opt.num_date_diff_classes,)
        elif opt.model_depth == 101:
            model = longi_models.resnet101(
                in_channel=opt.input_channels,
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_date_diff_classes=opt.num_date_diff_classes,)
        elif opt.model_depth == 152:
            model = longi_models.resnet152(
                in_channel=opt.input_channels,
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_date_diff_classes=opt.num_date_diff_classes,)
        elif opt.model_depth == 200:
            model = longi_models.resnet200(
                in_channel=opt.input_channels,
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_date_diff_classes=opt.num_date_diff_classes,)
    
    if not opt.no_cuda:
        print ('using cuda')
        print ('gpu:', opt.gpu)

        if opt.multiprocessing_distributed:
            model = model.cuda() 
            model = nn.DataParallel(model, device_ids=opt.gpu)
            net_dict = model.state_dict() 
        else:
            import os
            os.environ["CUDA_VISIBLE_DEVICES"]=str(opt.gpu)
            model = model.cuda() 
            model = nn.DataParallel(model, device_ids=None)
            net_dict = model.state_dict()
    else:
        net_dict = model.state_dict()

    if opt.pretrain_path:
        print ('loading pretrained model {}'.format(opt.pretrain_path))
        pretrain_file = "{}/{}_{}.pth".format(opt.pretrain_path, opt.model, opt.model_depth)

        pretrain = torch.load(pretrain_file)
        pretrain['state_dict'].popitem(last=False) # pop first convolutional layer when using pretrained data from MedicalNet

        for i in range(16):
            pretrain['state_dict'].popitem(last=True) # pop last convolutional layer when using pretrained data from label = 0 (before/after test)

        pretrain['state_dict'].popitem(last=True) # pop fc.weights, fc.bias

        pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}

        net_dict.update(pretrain_dict)
        model.load_state_dict(net_dict)

        new_parameters = []
        for pname, p in model.named_parameters():
            for layer_name in ['conv_seg']: # was args.new_layer_names in the past
                if pname.find(layer_name) >= 0:
                    new_parameters.append(p)
                    break

    return model

