import torch
from torch import optim
from torchtext import data
import dataset
import torch.nn as nn
import torch.nn.functional as F


def binary_accuracy(preds, y):
    """
    :param preds:
    :param y:
    :return: accuracy per batch, i.e if you get 8/10 right, this returns 0.8
    """
    rounded_preds = torch.round(torch.sigmoid(preds))
    # round predictions to the closest integer
    correct = (rounded_preds == y).float()
    # convert into float for division
    acc = correct.sum() / len(correct)
    return acc


class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(fs, embedding_dim)) for fs in filter_sizes])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x = [sent len, batch size]
        x = x.permute(1, 0)
        # x = [batch size, sent len]
        embedded = self.embedding(x)
        # embedded = [batch size, sent len, emb dim]
        embedded = embedded.unsqueeze(1)
        # embedded = [batch size, 1, sent len, emb dim]
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        # conv_n = [batch size, n_filters, sent len - filter_sizes[n]]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        # pooled_n = [batch size, n_filters]
        cat = self.dropout(torch.cat(pooled, dim=1))
        # cat = [batch size, n_filters * len(filter_sizes)]
        return self.fc(cat)


def train_(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for batch in iterator:
        optimizer.zero_grad()
        predictions = model(batch.text).squeeze(1)
        loss = criterion(predictions, batch.label)
        acc = binary_accuracy(predictions, batch.label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text).squeeze(1)
            loss = criterion(predictions, batch.label)
            acc = binary_accuracy(predictions, batch.label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def dump_tokenizer(text):
    tokens = text.split(' ')
    if len(tokens) < 5:
        for i in range(0, 5 - len(tokens)):
            tokens.append('<PAD>')
    return tokens


if __name__ == '__main__':
    # initialize
    SEED = 14
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # create dataset
    TEXT = data.Field(tokenize=dump_tokenizer)
    LABEL = data.LabelField(tensor_type=torch.FloatTensor)
    fields = [('text', TEXT), ('label', LABEL)]
    paths = {'train': 'train.csv', 'test': 'test.csv'}
    train, valid, test = dataset.get_dataset(fields=fields, paths=paths,
                                             view_info=True, valid_seed=SEED)
    # create vocabulary
    TEXT.build_vocab(train, max_size=2500)
    LABEL.build_vocab(train)

    batch_size = 64
    train_iter, valid_iter, test_iter = data.BucketIterator.splits(
        (train, valid, test),
        batch_size=batch_size,
        sort_key=lambda x: len(x.text),
        repeat=False
    )

    input_dim = len(TEXT.vocab)
    embedding_dim = 16
    n_filters = 32
    filter_sizes = [3, 4, 5]
    output_dim = 1
    dropout = 0.5
    model = CNN(input_dim, embedding_dim, n_filters,
                filter_sizes, output_dim, dropout)

    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()
    model = model.to(device)
    criterion = criterion.to(device)

    n_epochs = 500
    for epoch in range(n_epochs):
        train_loss, train_acc = train_(model, train_iter, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_iter, criterion)
        print(f'epoch: {epoch+1:02}\n'
              f'train loss: {train_loss:.3f}, train_acc: {train_acc*100:.2f}%, '
              f'val . loss: {valid_loss:.3f}, val . acc: {valid_acc*100:.2f}%'
        )
    test_loss, test_acc = evaluate(model, test_iter, criterion)
    print(f'test loss: {test_loss:.3f}, test acc: {test_acc*100:.2f}%')