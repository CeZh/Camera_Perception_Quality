import torch
import argparse, os, yaml, json
from core.builder.build_dataset import register_dataset
from core.builder.build_loss import loss_function
from core.builder.build_model import Perceptual_Quality_Estimation
from configs.configs import configuration_update
from tqdm import tqdm, trange
import logging
import logging.config
import time
from time import time, ctime
import warnings
import numpy as np
from sklearn import metrics
from scipy import stats
import random
import cv2
from PIL import Image
from utils import image_compression
warnings.filterwarnings("ignore")


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def loss_cal(mode, output, results_summary, eval_loss_functions, **kwargs):
    if mode == 'reg':
        if kwargs.get('lds') == True and kwargs['lds']:
            reg_iou_loss = eval_loss_functions['regress_iou_loss'](output['regress'].squeeze()[0],
                                                                   kwargs['regress_label_iou'], kwargs['weights'][:, 0])
            reg_prob_loss = eval_loss_functions['regress_prob_loss'](output['regress'].squeeze()[1],
                                                                     kwargs['regress_label_prob'], kwargs['weights'][:, 1])
        else:
            reg_iou_loss = eval_loss_functions['regress_iou_loss'](output['regress'].squeeze()[0],
                                                                   kwargs['regress_label_iou'])
            reg_prob_loss = eval_loss_functions['regress_prob_loss'](output['regress'].squeeze()[1],
                                                                     kwargs['regress_label_prob'])
        reg_iou_loss_MAE = eval_loss_functions['regress_iou_loss_eval'](output['regress'].squeeze()[0],
                                                                        kwargs['regress_label_iou'])
        reg_prob_loss_MAE = eval_loss_functions['regress_prob_loss_eval'](output['regress'].squeeze()[1],
                                                                          kwargs['regress_label_prob'])
        results_summary['regress_iou_loss'].append(reg_iou_loss.item())
        results_summary['regress_prob_loss'].append(reg_prob_loss.item())
        results_summary['regress_iou_loss_MAE'].append(reg_iou_loss_MAE.item())
        results_summary['regress_prob_loss_MAE'].append(reg_prob_loss_MAE.item())
        results_summary['output_regress_iou'].append(output['regress'].squeeze()[0].item())
        results_summary['output_regress_prob'].append(output['regress'].squeeze()[1].item())
        results_summary['gt_regress_iou'].append(kwargs['regress_label_iou'].item())
        results_summary['gt_regress_prob'].append(kwargs['regress_label_prob'].item())
    elif mode == 'cls':
        cls_iou_loss = eval_loss_functions['classes_iou_loss'](output['class'][0], kwargs['class_label_iou'])
        cls_prob_loss = eval_loss_functions['classes_prob_loss'](output['class'][1], kwargs['class_label_prob'])
        _, output_class_iou = torch.max(output['class'][0], 1)
        _, output_class_prob = torch.max(output['class'][1], 1)
        results_summary['class_iou_loss'].append(cls_iou_loss.item())
        results_summary['class_prob_loss'].append(cls_prob_loss.item())
        results_summary['output_class_iou'].append(output_class_iou.item())
        results_summary['output_class_prob'].append(output_class_prob.item())
        results_summary['gt_class_iou'].append(kwargs['class_label_iou'].item())
        results_summary['gt_class_prob'].append(kwargs['class_label_prob'].item())
    else:
        reg_iou_loss = eval_loss_functions['regress_iou_loss'](output['regress'].squeeze()[0], kwargs['regress_label_iou'])
        reg_prob_loss = eval_loss_functions['regress_prob_loss'](output['regress'].squeeze()[1], kwargs['regress_label_prob'])
        reg_iou_loss_MAE = eval_loss_functions['regress_iou_loss_eval'](output['regress'].squeeze()[0],
                                                                        kwargs['regress_label_iou'])
        reg_prob_loss_MAE = eval_loss_functions['regress_prob_loss_eval'](output['regress'].squeeze()[1],
                                                                          kwargs['regress_label_prob'])

        cls_iou_loss = eval_loss_functions['classes_iou_loss'](output['class'][0], kwargs['class_label_iou'])
        cls_prob_loss = eval_loss_functions['classes_prob_loss'](output['class'][1], kwargs['class_label_prob'])
        _, output_class_iou = torch.max(output['class'][0], 1)
        _, output_class_prob = torch.max(output['class'][1], 1)


        results_summary['regress_iou_loss'].append(reg_iou_loss.item())
        results_summary['regress_prob_loss'].append(reg_prob_loss.item())
        results_summary['regress_iou_loss_MAE'].append(reg_iou_loss_MAE.item())
        results_summary['regress_prob_loss_MAE'].append(reg_prob_loss_MAE.item())
        results_summary['output_regress_iou'].append(output['regress'].squeeze()[0].item())
        results_summary['output_regress_prob'].append(output['regress'].squeeze()[1].item())
        results_summary['gt_regress_iou'].append(kwargs['regress_label_iou'].item())
        results_summary['gt_regress_prob'].append(kwargs['regress_label_prob'].item())


        results_summary['class_iou_loss'].append(cls_iou_loss.item())
        results_summary['class_prob_loss'].append(cls_prob_loss.item())
        results_summary['output_class_iou'].append(output_class_iou.item())
        results_summary['output_class_prob'].append(output_class_prob.item())
        results_summary['gt_class_iou'].append(kwargs['class_label_iou'].item())
        results_summary['gt_class_prob'].append(kwargs['class_label_prob'].item())

    return results_summary

def class_acc(results_summary):
    pred_iou, pred_prob = results_summary['output_class_iou'], results_summary['output_class_prob']
    gt_iou, gt_prob = results_summary['gt_class_iou'], results_summary['gt_class_prob']


    iou_acc = metrics.balanced_accuracy_score(gt_iou, pred_iou)
    prob_acc = metrics.balanced_accuracy_score(gt_prob, pred_prob)
    accuracy = [iou_acc, prob_acc]
    results_summary['iou_accuracy'] = iou_acc
    results_summary['prob_accuracy'] = prob_acc
    return results_summary, accuracy

def regress_acc(results_summary):
    reg_iou, reg_prob = results_summary['output_regress_iou'], results_summary['output_regress_prob']
    gt_regress_iou, gt_regress_prob = results_summary['gt_regress_iou'], results_summary['gt_regress_prob']
    results_summary['srcc_iou'], _ = stats.spearmanr(reg_iou, gt_regress_iou)
    results_summary['plcc_iou'], _ = stats.pearsonr(reg_iou, gt_regress_iou)
    results_summary['srcc_prob'], _ = stats.spearmanr(reg_prob, gt_regress_prob)
    results_summary['plcc_prob'], _ = stats.pearsonr(reg_prob, gt_regress_prob)
    results_summary['r2_iou'] = metrics.r2_score(reg_iou, gt_regress_iou)
    results_summary['r2_prob'] = metrics.r2_score(reg_prob, gt_regress_prob)

    return results_summary

def main(args):
    if args.video:
        files = os.listdir(args.file_dir)
    else:
        files = os.listdir(args.file_dir)
    with open(args.configs, 'r') as yamlfile:
        configs = yaml.load(yamlfile, Loader=yaml.FullLoader)

    configs = configuration_update(configs, args)

    # check gpu availability
    if torch.cuda.is_available:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    log_path = args.log_path
    model_path = args.model_path
    logging.basicConfig(filename=os.path.join(log_path, str(configs['dataset_parameters']['dataset_name']) + '_demo.log'),
                        filemode='w',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level='NOTSET')
    # logging.config.dictConfig(configs)
    logger = logging.getLogger(__name__)
    logger.info('start evaluating...')
    logger.info(['Running Date: ', ctime(time())])
    logger.info(['Multi-Scale:', args.scale])


    model = Perceptual_Quality_Estimation(configs)
    model = model.to(device)

    model.load_state_dict(torch.load(os.path.join(args.model_path, 'best_loss_total.pt')))
    model.eval()
    iter_val = 0

    results_summary = {'output': [],
                       'names': []}
    with torch.no_grad():
        with trange(len(files), unit="iteration", desc='iteration ' + str(iter)) as pbar:
            for idx, content in enumerate(files):
                image = pil_loader(os.path.join(args.file_dir, content))
                image = image_compression.transform_val(image, 512, super_pixel=configs['superpixel_parameters'])
                super_pixel = image['x'].to('cuda')
                super_pos = image['pos'].to('cuda')
                output = model(image['img_super'].unsqueeze(0).to('cuda'), super_pixel=super_pixel.unsqueeze(0), super_pos=super_pos)
                # output = model(image.unsqueeze(0).to('cuda'))
                output = output['regress']
                img = cv2.imread(os.path.join(args.file_dir, content))
                img = cv2.putText(img, 'Perceptual Quality: ', org=(25, 50), color = (125, 0, 125), thickness=2, fontScale=2, fontFace=cv2.LINE_AA)
                img = cv2.putText(img, str(output.item()), org = (800, 50), color = (125, 0, 125), thickness=2, fontScale=2, fontFace=cv2.LINE_AA)
                save_name = os.path.join(args.model_path, args.save_folder, content)
                if not os.path.exists(os.path.split(save_name)[0]):
                    os.mkdir(os.path.split(save_name)[0])
                cv2.imwrite(save_name, img)
                # cv2.imshow('image', img)
                # cv2.waitKey()
                # cv2.destroyAllWindows()
                pbar.set_postfix(perceptual_quality=output.item())
                pbar.n += 1

        with open(os.path.join(model_path, 'demo_results.json'), 'w') as f:
            json.dump(results_summary, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training for Image Perceptual Quality')
    parser.add_argument('--configs', default='./configs/bdd100k/super_vit_linear_superdim_12.yaml', help='Configuration file for dataset')
    parser.add_argument('--log_path', default='./runs/super_vit_base/logging_files')
    parser.add_argument('--model_path', default = './runs/super_vit_base/models')
    parser.add_argument('--scale', default = 3, type = int, help='Multi-Scale Value')
    parser.add_argument('--dropout_rate', default=0.2, type=float, help='Dropout rate for mlp head')
    parser.add_argument('--attention_mlp', default=4096, type=int, help='Attention Module Linear Dimension')
    parser.add_argument('--file_dir', default= '/media/ce-zhang/Automated_Driving_Dataset/Customize_Dataset/ASIM_Dataset/GoPro_Version/Dec_13_Loop_Afternoon_Sunny/Images')
    parser.add_argument('--video', default=False, type=bool, help='Whether the input files are video or image')
    parser.add_argument('--save_folder', default='Dec_13_Loop_Afternoon_Sunny', type=str, help='Folder name')
    args = parser.parse_args()
    main(args)