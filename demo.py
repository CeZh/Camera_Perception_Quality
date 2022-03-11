import torch
import argparse, os, yaml, json
from core.builder.build_model import Perceptual_Quality_Estimation
from configs.configs import configuration_update
from tqdm import tqdm, trange
import logging
import logging.config
import time
from time import time, ctime
import warnings
import cv2
from PIL import Image
from utils import image_compression
warnings.filterwarnings("ignore")


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def main(args):
    files = os.listdir(args.file_dir)
    with open(args.configs, 'r') as yamlfile:
        configs = yaml.load(yamlfile, Loader=yaml.FullLoader)

    configs = configuration_update(configs, args)

    # check gpu availability
    if torch.cuda.is_available:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    log_path = 'demo_logger'
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    if not os.path.exists(os.path.join(args.output, log_path)):
        os.mkdir(os.path.join(args.output, log_path))

    logging.basicConfig(filename=os.path.join(args.output, log_path, str(configs['dataset_parameters']['dataset_name']) + '_demo.log'),
                        filemode='w',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level='NOTSET')
    logger = logging.getLogger(__name__)
    logger.info('start evaluating...')
    logger.info(['Running Date: ', ctime(time())])


    model = Perceptual_Quality_Estimation(configs)


    model.load_state_dict(torch.load(os.path.join(args.model_path, 'best_loss_total.pt')))
    model.eval()
    model = model.to(device)

    with torch.no_grad():
        with trange(len(files), unit="iteration", desc='iteration ' + str(iter)) as pbar:
            for idx, content in enumerate(files):
                image = pil_loader(os.path.join(args.file_dir, content))
                image = image.resize([configs['superpixel_parameters']['original_width'],
                                      configs['superpixel_parameters']['original_height']], Image.ANTIALIAS)
                image = image_compression.transform_val(image, 512, super_pixel=configs['superpixel_parameters'])
                super_pixel = image['x'].to(device)
                super_pos = image['pos'].to(device)
                output = model(image['img_super'].unsqueeze(0).to(device), super_pixel=super_pixel.unsqueeze(0).to(device), super_pos=super_pos.unsqueeze(0).to(device))
                # output = model(image.unsqueeze(0).to('cuda'))
                output = output['regress']
                img = cv2.imread(os.path.join(args.file_dir, content))
                img = cv2.putText(img, 'Perceptual Quality: ', org=(25, 50), color = (125, 0, 125), thickness=2, fontScale=2, fontFace=cv2.LINE_AA)
                img = cv2.putText(img, str(output.item()), org = (800, 50), color = (125, 0, 125), thickness=2, fontScale=2, fontFace=cv2.LINE_AA)
                save_name = os.path.join(args.output, content)
                if not os.path.exists(os.path.split(save_name)[0]):
                    os.mkdir(os.path.split(save_name)[0])
                cv2.imwrite(save_name, img)
                pbar.set_postfix(perceptual_quality=output.item())
                pbar.n += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training for Image Perceptual Quality')
    parser.add_argument('--configs', default='./configs/bdd100k/super_vit_linear.yaml', help='Configuration file for dataset')
    parser.add_argument('--model_path', default = './model_weights')
    parser.add_argument('--file_dir', default= './demo_images')
    parser.add_argument('--output', default='demo_outputs', type=str, help='Folder name')
    args = parser.parse_args()
    main(args)