import os.path
import sys

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

import model
from torch import optim, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from dataset import CocoLocalizationDataset

class Trainer:
    def __init__(self,
                 model,
                 optimizer,
                 criterion,
                 writer,
                 config,
                 datasets: dict = None,
                 dataloaders: dict = None,
                 transform=None,
                 target_transform=None,
                 metrics=None):

        self.config = config
        self.device = self.config.device
        self.model = model

        self.optimizer = optimizer
        self.criterion = criterion

        self.datasets = datasets if not None else dict()
        self.dataloaders = dataloaders if not None else dict()

        self.transform = transform
        self.target_transform = target_transform

        self.metrics = metrics

        self.writer = writer

        self._global_step = dict()

        self._loss_train_step = 0
        self._loss_eval_step = 0

    def _get_global_step(self, data_type):
        self._global_step[data_type] = -1

    def _get_data(self, data_type):
        transform = self.transform[data_type] if self.transform is not None else None
        target_transform = self.target_transform[data_type] if self.target_transform is not None else None

        self.datasets[data_type] = CocoLocalizationDataset(root=os.path.join(self.config.dataset_path, data_type),
                                                           annFile=os.path.join(self.config.dataset_path, 'annotations',
                                                                                f'instances_{data_type}.json'),
                                                           transform=transform,
                                                           target_transform=target_transform)

        self.dataloaders[data_type] = DataLoader(self.datasets[data_type], batch_size=self.config.batch_size,
                                                 shuffle=self.config.shuffle)

    def _epoch_step(self, stage = 'test', epoch = None):

        if stage not in self.dataloaders:
            self._get_data(stage)

        if stage not in self._global_step:
            self._get_global_step(stage)

        if stage == 'train':
            self.model.train()
            self._loss_train_step = 0

        else:
            self.model.eval()
            self._loss_eval_step = 0


        for step, (images, targets) in enumerate(self.dataloaders[stage]):

            self._global_step[stage] += 1

            predictions = self.model(images.to(self.device))

            if stage == 'train':
                self.optimizer.zero_grad()

            loss = self.criterion(predictions, targets.to(self.device))

            self.writer.add_scalar(f'{stage}/loss', loss, self._global_step[stage])


            if stage == 'train':
                loss.backward()
                self.optimizer.step()
                # self.scheduler.step(loss.item())
                self._loss_train_step += loss.item()
                running_loss = self._loss_train_step / (step + 1)

            if self.config.debug and step % self.config.show_each == 0:
                self._print_overwrite(step, (epoch + 1) * step, loss, stage)

    def fit(self, epoch_num):
        return self._epoch_step(stage='train', epoch=epoch_num)

    @torch.no_grad()
    def validation(self, i_epoch):
        self._epoch_step(stage='val', epoch=i_epoch)

    @torch.no_grad()
    def test(self):
        self._epoch_step(stage='test')

    def _print_overwrite(self, step, total_step, loss, stage):
        sys.stdout.write('\r')
        if stage == 'train':
            sys.stdout.write("Train Steps: %d/%d  Loss: %.4f " % (step, total_step, loss))
            # sys.stdout.write("Train Steps: %d/%d  Acc: %.4f " % (step, total_step, loss))
        else:
            sys.stdout.write("Valid Steps: %d/%d  Loss: %.4f " % (step, total_step, loss))

        sys.stdout.flush()


# class Trainer:
#     def __init__(self,
#                  model, optimizer,
#                  scheduler, criterion,
#                  writer, cfg,
#                  datasets_dict = None,
#                  dataloaders_dict = None,
#                  transform: dict = None,
#                  target_transform: dict = None,
#                  metrics: list = None):
#         self.cfg = cfg
#         self.model = model
#
#         self.optimizer = optimizer
#         self.scheduler = scheduler
#         self.criterion = criterion
#         self.metrics = metrics
#         self.transform = transform
#         self.target_transform = target_transform
#         self.device = self.cfg.device
#         self.writer = writer
#
#         self.datasets = datasets_dict if datasets_dict is not None else dict()
#         self.dataloaders = dataloaders_dict if dataloaders_dict is not None else dict()
#
#         self._global_step = dict()
#
#     def _get_global_step(self, data_type):
#         self._global_step[data_type] = -1
#
#     def _get_data(self, data_type):
#         transform = self.transform[data_type] if self.transform is not None else None
#         target_transform = self.target_transform[data_type] if self.target_transform is not None else None
#
#         self.datasets[data_type] = CocoLocalizationDataset(root=os.path.join(self.cfg.DATASET_PATH, data_type),
#                                                            annFile=os.path.join(self.cfg.DATASET_PATH, 'annotations',
#                                                                                 f'instances_{data_type}.json'),
#                                                            transform=transform,
#                                                            target_transform=target_transform)
#
#         self.dataloaders[data_type] = DataLoader(self.datasets[data_type], batch_size=self.cfg.batch_size,
#                                                  shuffle=self.cfg.shuffle)
#
#         # def collate_target_dict(batch):
#         #     img_list, target_list = batch
#         #     target = target_list[0]
#         #     for t in target_list[1:]:
#         #         for k, v in t.items():
#         #             target[k] = target[k]
#         #     return torch.stack(img_list, 0), target
#
#         # self.dataloaders[data_type] = DataLoader(self.datasets[data_type], batch_size=self.cfg.batch_size,
#         #                                          collate_fn=collate_target_dict, shuffle=self.cfg.shuffle)
#
#     @torch.no_grad()
#     def _calc_epoch_metrics(self, stage):
#         self._calc_metrics(stage, self.cfg.debug, is_epoch=True)
#
#     @torch.no_grad()
#     def _calc_batch_metrics(self, masks, targets, stage, debug):
#         self._calc_metrics(stage, debug, one_hot_argmax(masks), targets)
#
#     def _calc_metrics(self, stage, debug, *batch, is_epoch: bool = False):
#         for metric in self.metrics:
#             values = metric(is_epoch, *batch).tolist()
#             metric_name = type(metric).__name__
#
#             for cls, scalar in (zip(self.classes, values) if hasattr(self, 'classes') else enumerate(values)):
#                 self.writer.add_scalar(f'{stage}/{metric_name}/{cls}', scalar, self._global_step[stage])
#
#             self.writer.add_scalar(f'{stage}/{metric_name}/overall',
#                                    sum(values) / len(values), self._global_step[stage])
#
#             if debug:
#                 print("{}: {}".format(metric_name, values))
#
#     def _epoch_step(self, stage='test', epoch=None):
#
#         if stage not in self.dataloaders:
#             self._get_data(stage)
#
#         if stage not in self._global_step:
#             self._get_global_step(stage)
#
#         calc_metrics = self.metrics is not None and self.metrics
#         print('\n_______', stage, f'epoch{epoch}' if epoch is not None else '',
#               'len:', len(self.dataloaders[stage]), '_______')
#
#         for i, (images, targets) in enumerate(self.dataloaders[stage]):
#
#             self._global_step[stage] += 1
#             debug = self.cfg.debug and i % self.cfg.show_each == 0
#
#             # one_hots = F.one_hot(targets, num_classes=self.cfg.out_features).transpose(1, -1).squeeze(-1)
#             predictions = self.model(images.to(self.device))
#
#             # if calc_metrics:
#             #     self._calc_batch_metrics(predictions, one_hots, stage, debug)
#
# #             loss = self.criterion(predictions, targets.to(self.device).type(torch.cuda.FloatTensor))
#             loss = self.criterion(predictions, targets.to(self.device))
#             self.writer.add_scalar(f'{stage}/loss', loss, self._global_step[stage])
#
#             if debug:
#                 print('\n___', f'Iteration {i}', '___')
#                 print(f'Train Loss: {loss.item()}')
#
#             if stage == 'train':
#                 self.optimizer.zero_grad()
#                 loss.backward()
#                 self.optimizer.step()
#                 self.scheduler.step(loss.detach())
#
#         if calc_metrics and epoch is not None:
#             print('\n___', f'Epoch Summary', '___')
#             self._calc_epoch_metrics(stage)
#
#     def fit(self, i_epoch):
#         self._epoch_step(stage='train', epoch=i_epoch)
#
#     @torch.no_grad()
#     def validation(self, i_epoch):
#         self._epoch_step(stage='val', epoch=i_epoch)
#
#     @torch.no_grad()
#     def test(self):
#         self._epoch_step(stage='test')
#
#     def save_model(self, epoch, path=None):
#         path = self.cfg.SAVE_PATH if path is None else path
#
#         if not os.path.exists(path):
#             os.makedirs(path)
#
#         path = os.path.join(path, f'{epoch}.pth')
#
#         checkpoint = dict(epoch=self._global_step,
#                           model=self.model.state_dict(),
#                           optimizer=self.optimizer.state_dict())
#
#         torch.save(checkpoint, path)
#         print('model saved, epoch:', epoch)
#
#     def load_model(self, path):
#         checkpoint = torch.load(path)
#         self._global_step = checkpoint['epoch']
#         self.model.load_state_dict(checkpoint['model'])
#         self.optimizer.load_state_dict(checkpoint['optimizer'])
#         print('model loaded')
