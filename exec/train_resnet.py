import json
import os.path
import pickle

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

import model
from torch import optim, nn
from torchvision import transforms, datasets

from dataset import CocoLocalizationDataset
from exec.trainer import Trainer
# from transforms import UnNormalize, GetFromAnns, ClassMapping
from config import Config
# from metrics import MeanIoU, BalancedAccuracy
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':

    # ROOT = ''
    DATASET_ROOT = '../data/COCO/'

    cfg = Config(dataset_path=DATASET_ROOT,
                 dataset_name='COCO_SOLO', num_classes=80,
                 model_name='Resnet50', device='cpu',
                 batch_size=64, lr=0.001, overfit=False,
                 debug=True, show_each=1,
                 seed=None)

    train_key, valid_key, test_key = 'train', 'val', 'test'

    if cfg.seed is not None:
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed(cfg.seed)

    resnet_preprocess_block = [transforms.RandomResizedCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],  # resnet imagnet
                                                    std=[0.229, 0.224, 0.225])]

    train_preprocess = transforms.Compose(resnet_preprocess_block)
    eval_preprocess = transforms.Compose(resnet_preprocess_block)

    # target_preprocess = transforms.Compose([GetFromAnns(category=True)])
    # target_preprocess = transforms.Compose([ClassMapping(cfg.out_features)])
    target_preprocess = None

    transform = {train_key: train_preprocess, valid_key: eval_preprocess, test_key: eval_preprocess}
    target_transform = {train_key: target_preprocess, valid_key: target_preprocess, test_key: target_preprocess}

    mapping_path = os.path.join(DATASET_ROOT, 'annotations', 'mapping.json')
    datasets_dict = {train_key: CocoLocalizationDataset(root=os.path.join(cfg.dataset_path, train_key),
                                                        annFile=os.path.join(cfg.dataset_path, 'annotations',
                                                                             f'instances_{train_key}.json'),
                                                        mapping=mapping_path,
                                                        transform=transform[train_key],
                                                        target_transform=target_transform[train_key]),

                     valid_key: CocoLocalizationDataset(root=os.path.join(cfg.dataset_path, valid_key),
                                                        annFile=os.path.join(cfg.dataset_path, 'annotations',
                                                                             f'instances_{valid_key}.json'),
                                                        mapping=mapping_path,
                                                        transform=transform[valid_key],
                                                        target_transform=target_transform[valid_key])}

    if cfg.overfit:
        shuffle = False
        for dataset in datasets_dict.values():
            dataset.set_overfit_mode(cfg.batch_size)
    else:
        shuffle = True

    dataloaders_dict = {train_key: DataLoader(datasets_dict[train_key],
                                              batch_size=cfg.batch_size, shuffle=shuffle),
                        valid_key: DataLoader(datasets_dict[valid_key],
                                              batch_size=cfg.batch_size)}

    # metrics = [BalancedAccuracy(cfg.out_features)]
    model = model.resnet50(pretrained=True,
                            num_classes=cfg.num_classes).to(cfg.device)

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = ReduceLROnPlateau(optimizer)
    criterion = nn.CrossEntropyLoss()

    writer = SummaryWriter(log_dir=cfg.LOG_PATH)

    trainer = Trainer(datasets=datasets_dict, dataloaders=dataloaders_dict,
                      model=model, optimizer=optimizer,
                      criterion=criterion, writer=writer, config=cfg)

    epoch = 1
    for epoch in range(epoch):
        trainer.fit(epoch)

        trainer.writer.add_scalar(f'scheduler lr', trainer.optimizer.param_groups[0]['lr'], epoch)
        trainer.validation(epoch)
