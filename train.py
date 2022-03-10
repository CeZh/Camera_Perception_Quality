import torch
import argparse, os, yaml, shutil
from core.builder.build_dataset import register_dataset
from core.builder.build_loss import loss_function
from core.builder.build_optimizer import optimizer_fn
from core.builder.build_model import Perceptual_Quality_Estimation
from core.engines.trainer import trainer

from configs.configs import configuration_update

import logging
import time
import warnings

# image quality assessment
import random


warnings.filterwarnings("ignore")


def convert_to_preferred_format(sec):
   sec = sec % (24 * 3600)
   hour = sec // 3600
   sec %= 3600
   min = sec // 60
   sec %= 60
   return "%02d:%02d:%02d" % (hour, min, sec)

def main(args):

    start_time = time.time()
    # define a training seed for reproducibility
    torch.manual_seed(39)

    # load configuration file and update the parameters
    with open(args.configs, 'r') as yamlfile:
        configs = yaml.load(yamlfile, Loader=yaml.FullLoader)

    configs = configuration_update(configs, args)

    # check gpu availability
    if torch.cuda.is_available:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    # create running folder
    run_path = args.run
    if args.resume == False:
        print('training a new model...')
        if os.path.isdir(run_path) is False:
            os.makedirs(run_path)

        if args.running_name == 'default':
            from datetime import datetime
            now = datetime.now()
            parent_path = os.path.join(run_path, args.running_name+'-'+now.strftime("%b-%d-%Y-%H:%M:%S"))
        else:
            parent_path = os.path.join(run_path, args.running_name)
        if os.path.exists(parent_path):
            shutil.rmtree(parent_path)
        os.mkdir(parent_path)
        log_path = os.path.join(parent_path, 'logging_files')
        model_path = os.path.join(parent_path, 'models')
        os.mkdir(log_path)
        os.mkdir(model_path)
    else:
        print('resuming from last training...')
        if args.running_name == 'default':
            from datetime import datetime
            now = datetime.now()
            parent_path = os.path.join(run_path, args.running_name+'-'+now.strftime("%b-%d-%Y-%H:%M:%S"))
        else:
            from datetime import datetime
            now = datetime.now()
            parent_path = os.path.join(run_path, args.running_name)
            log_path = os.path.join(parent_path, 'logging_files'+'-'+now.strftime("%b-%d-%Y-%H:%M:%S"))
            model_path = os.path.join(parent_path, 'models')
            os.mkdir(log_path)

    print('save the logger to: '+ log_path)
    print('save the model to: '+ model_path)

    # create logger
    print('start training...')
    logging.basicConfig(filename=os.path.join(log_path, str(configs['dataset_parameters']['dataset_name'])+'.log'),
                        filemode='w',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level='NOTSET')
    logger = logging.getLogger(__name__)


    logger.info('start training...')
    logger.info([ 'Name:', args.running_name, 'Super_Module_Dimension:', args.super_att_dim,
                  'Attention Layer:', args.depth, 'Attention Head:', args.head,
                  'Super-Segments:', args.segments])


    time.sleep(2)
    print('loading dataset...')
    print(configs['dataset_parameters']['dataset_name'])
    logger.info('load dataset...')

    # load dataset
    dataset_name = register_dataset(configs['dataset_parameters']['dataset_name'])

    # BDD Dataset
    if dataset_name.__name__ == 'BDD_Dataset':
        from core.builder.build_dataset import collate_bdd
        train_dataset = dataset_name(configs['dataset_parameters']['train_image_path'],
                                     configs['dataset_parameters']['train_label_path'],
                                     mode='train',
                                     extra=configs)
        val_dataset = dataset_name(configs['dataset_parameters']['val_image_path'],
                                   configs['dataset_parameters']['val_label_path'],
                                   mode='val',
                                   extra=configs)


        train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                        batch_size=configs['dataset_parameters']['batch_size'],
                                                        collate_fn=collate_bdd,
                                                        num_workers=configs['dataset_parameters']['num_workers'],
                                                        shuffle=True)
        val_data_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                      batch_size=1,
                                                      collate_fn=collate_bdd,
                                                      num_workers=0,
                                                      shuffle=False)

    # KITTI Dataset
    if dataset_name.__name__ == 'KITTI_Dataset':
        from core.builder.build_dataset import collate_kitti
        train_dataset = dataset_name(configs['dataset_parameters']['train_image_path'],
                                     configs['dataset_parameters']['train_label_path'],
                                     mode='train',
                                     extra=configs)
        val_dataset = dataset_name(configs['dataset_parameters']['val_image_path'],
                                   configs['dataset_parameters']['val_label_path'],
                                   mode='val',
                                   extra=configs)

        train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                        batch_size=configs['dataset_parameters']['batch_size'],
                                                        collate_fn=collate_kitti,
                                                        num_workers=configs['dataset_parameters']['num_workers'],
                                                        shuffle=True)
        val_data_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                      batch_size=1,
                                                      collate_fn=collate_kitti,
                                                      num_workers=0,
                                                      shuffle=False)

    # NuScene Dataset
    if dataset_name.__name__ == 'NuScene_Dataset':
        from core.builder.build_dataset import collate_nuscene
        train_dataset = dataset_name(configs['dataset_parameters']['train_image_path'],
                                     configs['dataset_parameters']['train_label_path'],
                                     mode='train',
                                     extra=configs)
        val_dataset = dataset_name(configs['dataset_parameters']['val_image_path'],
                                   configs['dataset_parameters']['val_label_path'],
                                   mode='val',
                                   extra=configs)

        train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                        batch_size=configs['dataset_parameters']['batch_size'],
                                                        collate_fn=collate_nuscene,
                                                        num_workers=configs['dataset_parameters']['num_workers'],
                                                        shuffle=True)
        val_data_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                      batch_size=1,
                                                      collate_fn=collate_nuscene,
                                                      num_workers=0,
                                                      shuffle=False)

    # loss function and optimizers
    print('load loss function, model, and optimizer...')
    regress_loss = loss_function(configs['training_parameters']['loss_type'],
                                 balance=configs['dataset_parameters']['use_lds'])
    eval_regress_loss = loss_function(configs['training_parameters']['loss_type'], balance=False)
    eval_mae_loss = loss_function(configs['training_parameters']['eval_loss_type'], balance=False)
    train_loss_functions = {'regress_loss': regress_loss}

    eval_loss_functions = {'regress_loss': eval_regress_loss, 'regress_loss_eval': eval_mae_loss}

    model = Perceptual_Quality_Estimation(configs)
    if args.resume:
        print('loading from last epoch...')
        checkpoint = torch.load(args.resume_path)
        model.load_state_dict(checkpoint['model_parameters'])
        print('parameters successfully loaded!')
    else:
        if configs['model_parameters']['backbone'] == 'DarkNet_53':
            checkpoints = torch.load('./pretrain_model/model_best.pth.tar')
            backbone_model = model.backbone_module.state_dict()
            for name, param in checkpoints['state_dict'].items():
                if name not in backbone_model:
                    print(name+' is not in the new model!')
                    continue
                if name in backbone_model:
                    backbone_model[name] = checkpoints['state_dict'][name]
                    print(name+' is loaded!')
            model.backbone_module.load_state_dict(backbone_model)
    model = model.to(device)
    optimizer = optimizer_fn(configs['training_parameters']['optimizer'], model)

    # training and evaluation
    logger.info('training sample size is '+str(len(train_data_loader)*configs['dataset_parameters']['batch_size']))
    best_loss = trainer(configs, model, train_data_loader,
                        optimizer, device, logger, train_loss_functions,
                        val_data_loader, eval_loss_functions, model_path,
                        args)
    print('Training Done!')
    print('Best Regression Accuracy: ' + str(best_loss))
    span_time = convert_to_preferred_format(time.time() - start_time)
    print('Training Time: ' + str(span_time))
    logger.info('############Training Finished!##############')
    logger.info(['Total training time is', span_time])




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training for Image Perceptual Quality')
    parser.add_argument('--configs', default='./configs/bdd100k/super_vit_linear.yaml', help = 'Configuration file for dataset')
    parser.add_argument('--run', default='runs/bdd_dataset', help='Running directory for model training')
    parser.add_argument('--running_name', default='super_vit', type = str, help='Running model features')
    parser.add_argument('--super_att_dim', default=18, type= int, help='Super Pixel Module Feature Dimension')
    parser.add_argument('--depth', default = 6, type = int, help='Pixel and Super Pixel Attention Layer Depth')
    parser.add_argument('--head', default=8, type=int, help='Pixel and Super Pixel Attention Layer Head')
    parser.add_argument('--segments', default=500, type=int, help='Segmentation pieces')
    parser.add_argument('--resume', default=False, type=bool, help='whether to resume the training')
    parser.add_argument('--resume_path', default='runs/nuscene_dataset/bilinear/models/checkpoint.pt', type=str, help='Load from checkpoint, usually named as checkpoint.pt')
    args = parser.parse_args()
    main(args)