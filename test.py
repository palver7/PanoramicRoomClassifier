from efficientnet_pytorch.model import EfficientNet
import argparse
import logging
#import sagemaker_containers
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.models
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
#import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def _test(args):
    """
    is_distributed = len(args.hosts) > 1 and args.dist_backend is not None
    logger.debug("Distributed training - {}".format(is_distributed))

    if is_distributed:
        # Initialize the distributed environment.
        world_size = len(args.hosts)
        os.environ['WORLD_SIZE'] = str(world_size)
        host_rank = args.hosts.index(args.current_host)
        os.environ['RANK'] = str(host_rank)
        dist.init_process_group(backend=args.dist_backend, rank=host_rank, world_size=world_size)
        logger.info(
            'Initialized the distributed environment: \'{}\' backend on {} nodes. '.format(
                args.dist_backend,
                dist.get_world_size()) + 'Current host rank is {}. Using cuda: {}. Number of gpus: {}'.format(
                dist.get_rank(), torch.cuda.is_available(), args.num_gpus))
                
    """   
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info("Device Type: {}".format(device))

    logger.info("Loading dataset from ImageFolder")
    img_size = EfficientNet.get_image_size(args.model_name) 
    transform = transforms.Compose(
        [transforms.Resize((img_size,img_size)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    #target_transform = transforms.Compose([transforms.Resize((224,224)),
    #                                       transforms.ToTensor()])     

    root = os.path.join(args.data_dir,"val") 
    testset = torchvision.datasets.ImageFolder(root,transform = transform, target_transform = None)
    test_loader = DataLoader(testset, batch_size=args.batch_size,
                                               shuffle=False, num_workers=args.workers)
    class_map = testset.classes
    num_classes = len(testset.classes)                                          
    
    logger.info("Model loaded")
    model = model_fn(args.model_dir,args.model_name,num_classes,args.conv_type)
    
    with torch.no_grad():
        running_acc = 0.0
        for i, data in enumerate(test_loader):
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            model.eval()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            # print statistics
            running_acc += torch.sum(preds == labels.data)
        
        epoch_acc = running_acc.double() / len(testset)
        print("test Acc: %.3f" %(epoch_acc))
            
        """
        preds = torch.topk(outputs, k=7).indices.squeeze(0).tolist()  
        print(preds)      
        print('-----')
        if args.batch_size > 1:
            for batch in preds:
                for idx in batch:
                    category = class_map[idx]
                    prob = torch.softmax(outputs, dim=1)[0, idx].item()
                    print('{:<75} ({:.2f}%)'.format(category, prob*100))
        else:            
            for idx in preds:
                category = class_map[idx]
                prob = torch.softmax(outputs, dim=1)[0, idx].item()
                print('{:<75} ({:.2f}%)'.format(category, prob*100))
        """        
        
    print('Finished Testing')
    


def _save_model(model, model_dir):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, 'model.pth')
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.cpu().state_dict(), path)


def model_fn(model_dir,model_name,num_classes,conv_type):
    logger.info('model_fn')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if conv_type == 'Std':
        layerdict, offsetdict = None, None
    elif conv_type == 'Equi':
        layerdict, offsetdict = torch.load('layertest.pt'), torch.load('offsettest.pt')    
    model = EfficientNet.from_pretrained(model_name,conv_type=conv_type, layerdict=layerdict, offsetdict=offsetdict)
    #for param in model.parameters():
    #    param.requires_grad = False
    num_features = model._fc.in_features
    model._fc = nn.Linear(num_features,num_classes)
    if torch.cuda.device_count() > 1:
        logger.info("Gpu count: {}".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
    return model.to(device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--workers', type=int, default=2, metavar='W',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--batch_size', type=int, default=1, metavar='BS',
                        help='batch size (default: 1)')
    parser.add_argument('--model-dir', type=str, default="")
    parser.add_argument('--model-name', type=str,default="efficientnet-b0")
    parser.add_argument('--data-dir', type=str, default="")
    parser.add_argument('--conv_type', type=str,default="Std")
    
    #parser.add_argument('--dist_backend', type=str, default='gloo', help='distributed backend (default: gloo)')
    #env = sagemaker_containers.training_env()
    #parser.add_argument('--hosts', type=list, default=env.hosts)
    #parser.add_argument('--current-host', type=str, default=env.current_host)
    #parser.add_argument('--model-dir', type=str, default=env.model_dir)
    #parser.add_argument('--data-dir', type=str, default=env.channel_input_dirs.get('training'))
    #parser.add_argument('--num-gpus', type=int, default=env.num_gpus)

    _test(parser.parse_args())