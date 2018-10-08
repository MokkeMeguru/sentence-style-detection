import random

import torch
from torchtext import data


def get_dataset(fields: list, paths: dict,
                view_info=False, valid_seed=None):
    if len(fields) == 2:
        train, test = data.TabularDataset.splits(
            path='',
            train=paths['train'],
            test=paths['test'],
            format='csv',
            fields=fields,
            skip_header=False,
        )
        if view_info:
            print('train-length: {}\ntest-length: {}'
                  '\ntrain-example: {}\ntest-example: {}'
                  .format(len(train), len(test),
                          vars(train[0]), vars(test[0])))
        if valid_seed:
            new_train, valid = train.split(random_state=random.seed(valid_seed))
            return new_train, valid, test
        else:
            return train, test
    else:
        NotImplementedError()


if __name__ == '__main__':
    TEXT = data.Field()
    LABEL = data.LabelField(tensor_type=torch.FloatTensor)
    fields = [('text', TEXT), ('label', LABEL)]
    paths = {'train': 'train.csv', 'test': 'test.csv'}
    train, test = get_dataset(fields, paths, view_info=True)
