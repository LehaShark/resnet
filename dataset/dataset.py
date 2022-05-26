import json
from typing import Tuple, Any
from torchvision.datasets import CocoDetection

# class CocoLocalizationDataset(CocoDetection):
#     def __init__(self, root: str, annFile: str, transform=None, target_transform=None, transforms=None):
#         super(CocoLocalizationDataset, self).__init__(root, annFile, transform, target_transform, transforms)
#         self.overfit = False
#         self._batch_size = None
#
#     def set_overfit_mode(self, batch_size: int):
#         self.overfit = True
#         self._batch_size = batch_size
#
#     def unset_overfit_mode(self):
#         self.overfit = False
#
#     def __getitem__(self, index: int) -> Tuple[Any, Any]:
#         id = self.ids[index]
#         image = self._load_image(id)
#         target = self._load_target(id)[0]['category_id']
#
#         if self.transforms is not None:
#             image, mapped_target = self.transforms(image, target)
#
#         return image, target
#
#     def __len__(self) -> int:
#         return len(self.ids) if not self.overfit else self._batch_size


class CocoLocalizationDataset(CocoDetection):

    def __init__(self, root: str, annFile: str, mapping: str = None, transform=None, target_transform=None,
                 transforms=None):
        super().__init__(root, annFile, transform, target_transform, transforms)
        self.overfit = False
        self._batch_size = None
        self.mapping = None

        if mapping is not None:
            self._get_mapping(mapping)

    def _get_mapping(self, mapping_path):
        with open(mapping_path, 'r') as f:
            mapping = json.load(f)
            f.close()
        self.mapping = {int(k): v for k, v in mapping.items()}

    def set_overfit_mode(self, batch_size: int):
        self.overfit = True
        self._batch_size = batch_size

    def unset_overfit_mode(self):
        self.overfit = False

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)[0]['category_id']

        if self.mapping is not None:
            target = self.mapping[target]['mapped_id']

        if self.transforms is not None:
            image, mapped_target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self.ids) if not self.overfit else self._batch_size
