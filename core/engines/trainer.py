import time
import torch

from tqdm import tqdm, trange
from core.engines.evaluator import evaluator
import os, json
from scipy import stats
from utils import  pytorch_warmup
def loss_sum(criterion, iter_loss, output, targets):

    # document regress loss
    iter_loss['total_regression'] += criterion.item()
    iter_loss['output'].extend(output.squeeze().to('cpu').tolist())
    iter_loss['targets'].extend(targets.squeeze().to('cpu').tolist())
    return iter_loss

def loss_document(epoch_loss, loss_results, idx):

    # document regress loss
    epoch_loss['loss_total_regress'].append(loss_results['total_regression'] / idx)

    return epoch_loss


def trainer(configs, model, train_data_loader, optimizer,
            device, logger, loss_functions, val_data_loader,
            eval_loss_functions, model_path, args):
    epoch_loss = {'loss_total_regress': [],
                  'eval_loss_total_regress': []}
    iter = 0
    best_loss = 1e9
    if configs['training_parameters']['cosine_decay']:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               len(train_data_loader),
                                                               eta_min=2e-5)
        logger.info('Using Cosine Decay...')
        print('Using Cosine Decay...')

    if configs['training_parameters']['warm_up']:
        warmup_scheduler = pytorch_warmup.UntunedLinearWarmup(optimizer)
        logger.info('Warming up...')
        print('Warming up...')

    for epoch in range(configs['training_parameters']['epoch']):
        if args.resume:
            checkpoint = torch.load(args.resume_path)
            epoch = checkpoint['epoch_num']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print('optimizer parameters successfully loaded!')
            print('start to train at %d' % epoch)
            args.resume=False

        # start training
        iter_loss = {'total_regression': 0.0, 'output': [], 'targets': []}
        model.train()
        time.sleep(2.5)
        print('Start Training')
        with trange(len(train_data_loader), unit="iteration", desc='epoch '+str(epoch)) as pbar:
            for idx, content in enumerate(train_data_loader):
                # convert data to tensor
                targets = torch.tensor(content['labels']).to(device).float()
                image = torch.tensor(content['image']).to(device).float()
                optimizer.zero_grad()
                if configs['model_parameters']['use_superpixel']:
                    super_pixel = torch.tensor(content['superpixel']).to(device).float()
                    super_pos = torch.tensor(content['superpos']).to(device).int()
                    output = model(image, super_pixel = super_pixel, super_pos = super_pos)
                else:
                    output = model(image)

                # calculate loss
                if configs['dataset_parameters']['use_lds']:
                    weights = torch.tensor(content['weights']).to(device).float()
                    criterion = loss_functions['regress_loss'](output['regress'].squeeze(), targets, weights)
                else:
                    criterion = loss_functions['regress_loss'](output['regress'].squeeze(), targets)
                # backward
                criterion.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 100)
                optimizer.step()
                if configs['training_parameters']['cosine_decay']:
                    scheduler.step()
                if configs['training_parameters']['warm_up'] and epoch<=4:
                    warmup_scheduler.dampen()

                # sum loss
                iter_loss = loss_sum(criterion, iter_loss, output['regress'], targets)

                # log loss
                if idx % configs['training_parameters']['log_every'] == 0:
                    logger.info(['loss: ', str(criterion.item())])

                # tqdm
                pbar.set_postfix(loss=criterion.item(), learn_rate=scheduler.get_lr())
                pbar.n += 1

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_data_loader), eta_min=2e-5)
            # log loss for one epoch
            logger.info(['epoch:', epoch])
            logger.info(['epoch regression loss total: ', iter_loss['total_regression'] / idx])
            iter += 1
            # documentation
            epoch_loss = loss_document(epoch_loss, iter_loss, idx)
            train_srcc, _ = stats.spearmanr(iter_loss['output'], iter_loss['targets'])
            train_plcc, _ = stats.pearsonr(iter_loss['output'], iter_loss['targets'])
            logger.info(['train_srcc: ', train_srcc])
            logger.info(['train_plcc: ', train_plcc])

        results_summary = evaluator(configs, model, val_data_loader, device, logger, eval_loss_functions)
        eval_loss = (sum(results_summary['Eval_loss'])/len(results_summary['Eval_loss']))
        epoch_loss['eval_loss_total_regress'].append(eval_loss)

        path = 'checkpoint.pt'
        torch.save({
            'epoch_num': epoch+1,
            'model_parameters': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(model_path,path))
        logger.info('Checkpoint saved!')

        if eval_loss < best_loss:
            best_loss = eval_loss
            torch.save(model.state_dict(), os.path.join(model_path, 'best_loss_total.pt'))
            with open(os.path.join(model_path, 'best_eval_results_regress.json'), 'w') as file:
                json.dump(results_summary, file)

    with open(os.path.join(model_path, 'train_results.json'), 'w') as file:
        json.dump(epoch_loss, file)

    return best_loss