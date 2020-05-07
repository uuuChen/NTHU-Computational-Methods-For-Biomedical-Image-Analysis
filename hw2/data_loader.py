import os
import pandas as pd
import numpy as np
from torch.utils.data.dataset import Dataset
from pydicom import dcmread
import torch
import random
import torchvision.transforms as transforms


class CBIS_DDSM_Dataset(Dataset):
    def __init__(self, root_dir, dataset, method='mask', transform=None, data_balance=False):
        csv_path = os.path.join(root_dir, f'calc_case_description_{dataset}_set.csv')
        self.root_dir = root_dir
        self.dataset_dir_name = 'CBIS-DDSM_Dataset'
        self.transform = transform
        self.csv_df = pd.read_csv(csv_path)
        self.labelstr2int = {'BENIGN': 0, 'BENIGN_WITHOUT_CALLBACK': 0, 'MALIGNANT': 1}
        self.method2attrname = {'mask': 'cropped image file path', 'crop': 'ROI mask file path',
                                'ori': 'image file path'}
        self.attr_name = self.method2attrname[method]  # file path attribute name
        self.image_paths_with_labels = self._get_image_paths_with_labels(self.csv_df, balance=data_balance)

    def __getitem__(self, index):
        image_path, image_label = self.image_paths_with_labels[index]
        image_ds = dcmread(image_path)
        image_data = image_ds.pixel_array.astype(np.float32) / np.max(image_ds.pixel_array)
        if self.transform:
            image_data = self.transform(image_data)
            self.first_image = False
        # transforms.ToPILImage()(image_data).convert('LA').show()
        return image_data, image_label

    def __len__(self):
        return len(self.csv_df)

    def _get_image_paths_with_labels(self, df, balance):
        image_paths = list()
        label2imgpath = dict()
        for i in range(len(df)):
            label = self.labelstr2int[self.csv_df.loc[i, 'pathology']]
            csv_image_file_path_list = self.csv_df.loc[i, self.attr_name].split('/')
            image_dir_name, image_file_name = csv_image_file_path_list[0].strip(), csv_image_file_path_list[-1].strip()
            image_file_path = None
            for dirPath, dirNames, fileNames in os.walk(
                    os.path.join(self.root_dir, self.dataset_dir_name, image_dir_name)):
                if not dirNames:  # in the last directory
                    image_file_path = os.path.join(dirPath, image_file_name)
                    break
            if not image_file_path:
                raise Exception
            image_paths.append((image_file_path, label))
            label2imgpath.setdefault(label, list()).append(image_file_path)
        if balance:
            less_label, more_label = (0, 1) if len(label2imgpath[0]) < len(label2imgpath[1]) else (1, 0)
            diff_between_labels = len(label2imgpath[more_label]) - len(label2imgpath[less_label])
            less_image_paths = label2imgpath[less_label]
            for i in range(diff_between_labels):
                image_paths.append((less_image_paths[random.randint(0, len(less_image_paths)-1)], less_label))
        return image_paths


def get_numpy_data2d_with_labels(data_loader):
    data2d = None
    labels = list()
    for data, target in data_loader:
        batch_data2d = data.numpy()[:, 0, :, :].reshape(data.shape[0], -1)
        data2d = batch_data2d if data2d is None else np.vstack((data2d, batch_data2d))
        labels.extend(target.numpy())
    labels = np.array(labels)
    return data2d, labels






