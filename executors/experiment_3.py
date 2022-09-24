import os
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, LinearLR, CyclicLR
from warmup_scheduler import GradualWarmupScheduler

from configs import DatasetConfig
from configs import ModifyResNetConfig
from configs.trainer_config import TrainerConfig
from dataloader import ImageLoader
from executors.trainer import Trainer
from nets import ModifyResNet
from transforms.transforms import LabelSmoothing
from utils import get_weights, LinearStochasticDepth
from utils.utils import zero_weights_bn

if __name__ == '__main__':
    # DATASET_ROOT = 'data/'

    dataset_config = DatasetConfig()
    trainer_config = TrainerConfig()
    model_cfg = ModifyResNetConfig()

    keys = train_key, valid_key = 'train', 'valid'

    jitter_param = (0.6, 1.4)

    normalize = [transforms.ToTensor(),
                 transforms.Normalize(mean=[0.485, 0.456, 0.405],
                                      std=[0.229, 0.224, 0.225])]

    image_transforms = {train_key: transforms.Compose([transforms.RandomResizedCrop(224),
                                                       transforms.RandomHorizontalFlip(),
                                                       transforms.ColorJitter(brightness=jitter_param,
                                                                              saturation=jitter_param,
                                                                              hue=(-.2, .2)),
                                                       *normalize]),

                        valid_key: transforms.Compose([transforms.Resize(256),
                                                       transforms.CenterCrop(224),
                                                       *normalize])}

    target_transforms = {train_key: transforms.Compose([LabelSmoothing(12, smooth=0.1),
                                                        ])}


    datasets_dict = {k: ImageLoader(root=os.path.join(dataset_config.PATH, k),
                                    transform=image_transforms[k] if k in image_transforms else None,
                                    target_transform=target_transforms[k] if k in target_transforms else None)
                     for k in keys}


    dataloaders_dict = {train_key: DataLoader(datasets_dict[train_key],
                                              batch_size=trainer_config.batch_size, shuffle=True),
                        valid_key: DataLoader(datasets_dict[valid_key],
                                              batch_size=trainer_config.batch_size)}


    model = ModifyResNet(model_cfg, stochastic_depth=LinearStochasticDepth).to(trainer_config.device)

    # weight decay
    if trainer_config.weight_decay is not None:
        w, b = get_weights(model)
        params = [dict(params=w, weight_decay=trainer_config.weight_decay),
                  dict(params=b)]
    else:
        params = model.parameters()

    zero_weights_bn(model)

    optimizer = optim.SGD(params, lr=trainer_config.lr, weight_decay=trainer_config.weight_decay, momentum=trainer_config.momentum)
    criterion = nn.CrossEntropyLoss()

    writer = SummaryWriter(log_dir=trainer_config.LOG_PATH)

    epochs = trainer_config.epoch_num
    end_warmup = 4


    scheduler = CyclicLR(optimizer, base_lr=trainer_config.lr, max_lr=0.1)

    trainer = Trainer(dataloaders=dataloaders_dict,
                      model=model,
                      optimizer=optimizer,
                      criterion=criterion,
                      config=trainer_config,
                      writer=writer,
                      scheduler=scheduler)

    # epoch = 40
    for epoch in range(epochs + 1):
        # trainer.validation(epoch)
        #     trainer.fit(epoch)
        #     trainer.writer.add_scalar(f'scheduler lr', trainer.optimizer.param_groups[0]['lr'], epoch)
        trainer.fit(trainer_config.epoch_num)

        print('\n', '_______', epoch, '_______')
        if epoch % 5 == 0:
            trainer.save_model(epoch, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs'))


        if epoch % 10 == 0:
            trainer.validation(epoch)
