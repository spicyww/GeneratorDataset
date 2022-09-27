import numpy as np
import mindspore as ms
from mindspore import ops
from dataset.meta import Dataset
import mindspore.dataset as ds
import time
import mindspore.dataset.transforms.py_transforms as trans


class ytvos(Dataset):
    def __init__(self,
                 path,
                 split=None,
                 transform=None,
                 target_transform=None,
                 seq_mode=None,
                 batch_size=1,
                 repeat_num=1,
                 shuffle=False,
                 columns_list=['video', 'labels'],
                 num_parallel_workers=1,
                 shard_id=None,
                 num_shards=None
                 ):
        data = test_data()
        load_data = data.parse_dataset
        super().__init__(path=path,
                         split=split,
                         load_data=load_data,
                         batch_size=batch_size,
                         repeat_num=repeat_num,
                         shuffle=shuffle,
                         num_parallel_workers=num_parallel_workers,
                         num_shards=num_shards,
                         shard_id=shard_id,
                         resize=300,
                         transform=transform,
                         target_transform=target_transform,
                         mode=seq_mode,
                         columns_list=columns_list,
                         schema_json=None,
                         trans_record=None)

    def default_transform(self):
        trans = [default_trans()]
        return trans

    def pipelines(self):
        trans = self.default_transform()
        self.dataset = self.dataset.map(operations=trans,
                                        input_columns=self.columns_list,
                                        num_parallel_workers=self.num_parallel_workers)

    def run(self):
        """dataset pipeline"""
        self.pipelines()
        self.dataset = self.dataset.batch(self.batch_size, drop_remainder=True)
        self.dataset = self.dataset.repeat(self.repeat_num)
        return self.dataset


class default_trans(trans.PyTensorOperation):
    def __init__(self):
        self.cast = ops.Cast()

    def __call__(self, data, label):
        return data, label


class test_data():
    def __init__(self):
        self.data = np.random.randn(100, 36, 3, 2, 2).astype(np.float32)
        self.list = list(range(0, 100, 1))

    def parse_dataset(self, *args):
        if not args:
            return self.list, self.data
        print(args)
        return self.data[args], args


dataset_train = ytvos(path="/usr/dataset/VOS/",
                      split='train',
                      batch_size=1,
                      repeat_num=1,
                      shuffle=False)
dataset_train = dataset_train.run()
for data in dataset_train.create_tuple_iterator():
    input, idx = data
    print(input.shape)
