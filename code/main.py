from miscc.config import cfg, cfg_from_file
#from __future__ import print_function
import torch
import torchvision.transforms as transforms

import argparse
import os
import random
import sys
import pprint
import datetime
import dateutil.tz
import time
import pickle

#
from datasets import Dataset
from trainer import FineGAN_trainer as trainer
from trainer import FineGAN_evaluator as evaluator


if __name__ == "__main__":

    cfg_from_file("train.yml")

    cfg.GPU_ID = '0'

    if cfg.TRAIN.FLAG:
        print('Using config:')
        pprint.pprint(cfg)

    if not cfg.TRAIN.FLAG:
        manualSeed = 45
        '''
        Change this to have different random seed during evaluation
        '''
    else :
        manualSeed = random.randint(1, 10000)
    
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(manualSeed)

    # Evaluation part
    if not cfg.TRAIN.FLAG:
        algo = evaluator()
        algo.evaluate_finegan()

    # Training part
    else:
        now = datetime.datetime.now(dateutil.tz.tzlocal())
        timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
        output_dir = '/content/drive/My Drive/tocollab/output_finegan/%s_%s' % \
            (cfg.DATASET_NAME, timestamp)
        pkl_filename = 'cfg.pickle'

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(os.path.join(output_dir, pkl_filename), 'wb') as pk:
            pickle.dump(cfg, pk, protocol=pickle.HIGHEST_PROTOCOL)

        bshuffle = True

        # Get data loader
        imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM-1))
        image_transform = transforms.Compose([
            transforms.Scale(int(imsize * 76 / 64)),
            transforms.RandomCrop(imsize),
            transforms.RandomHorizontalFlip()])

        dataset = Dataset(cfg.DATA_DIR,
                          base_size=cfg.TREE.BASE_SIZE,
                          transform=image_transform)
        assert dataset
        num_gpu = len(cfg.GPU_ID.split(','))
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=cfg.TRAIN.BATCH_SIZE * num_gpu,
            drop_last=True, shuffle=bshuffle, num_workers=int(cfg.WORKERS))

        algo = trainer(output_dir, dataloader, imsize)

        start_t = time.time()
        algo.train()
        end_t = time.time()
        print('Total time for training:', end_t - start_t)
