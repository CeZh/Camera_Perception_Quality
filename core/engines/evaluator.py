import time
import torch
from tqdm import tqdm, trange

from sklearn import metrics
from scipy import stats
def loss_cal(output, targets, eval_loss_functions, results_summary):
    eval_loss = eval_loss_functions['regress_loss'](output.squeeze(), targets)
    l1_loss = eval_loss_functions['regress_loss_eval'](output.squeeze(), targets)

    results_summary['output'].append(output.item())
    results_summary['targets'].append(targets.item())
    results_summary['Eval_loss'].append(eval_loss.item())
    results_summary['L1_loss'].append(l1_loss.item())

    return results_summary

def evaluator(configs, model, val_data_loader, device, logger, eval_loss_functions):
    time.sleep(2.5)
    print('Start Evaluation...')
    model.eval()
    iter_val = 0
    results_summary = {'r2': 0.0, 'Eval_loss': [], 'L1_loss': [], 'output': [], 'targets': [], 'PLCC': [], 'SRCC': []}
    with torch.no_grad():
        with trange(len(val_data_loader), unit="iteration", desc='evaluation proces: ') as pbar:
            for idx, content in enumerate(val_data_loader):
                # image = transform(torch.tensor(content[0]).to(device)).float()
                targets = torch.tensor(content['labels']).to(device).float()
                image = torch.tensor(content['image']).to(device).float()
                if configs['model_parameters']['use_superpixel']:
                    super_pixel = torch.tensor(content['superpixel']).to(device).float()
                    super_pos = torch.tensor(content['superpos']).to(device).float()
                    output = model(image, super_pixel=super_pixel, super_pos=super_pos)
                else:
                    output = model(image)
                output = output['regress']
                results_summary = loss_cal(output, targets, eval_loss_functions, results_summary)

                pbar.set_postfix(total_loss=results_summary['Eval_loss'][idx])
                pbar.n += 1
            results_summary['r2'] = metrics.r2_score(results_summary['output'], results_summary['targets'])
            results_summary['SRCC'], _ = stats.spearmanr(results_summary['output'], results_summary['targets'])
            results_summary['PLCC'], _ = stats.pearsonr(results_summary['output'], results_summary['targets'])
            logger.info(['#############Evaluation Loss Results#############'])
            logger.info(['L1 Loss: ', sum(results_summary['L1_loss'])/len(results_summary['L1_loss'])])
            logger.info(['Eval Loss: ', sum(results_summary['Eval_loss'])/len(results_summary['Eval_loss'])])
            logger.info(['R2: ', results_summary['r2']])
            logger.info(['SRCC: ', results_summary['SRCC']])
            logger.info(['PLCC: ', results_summary['PLCC']])
    return results_summary