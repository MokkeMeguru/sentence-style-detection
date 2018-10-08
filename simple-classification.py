import torch
from torchtext import data
import dataset
import torch.nn as nn
import torch.optim as optim
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
    acc = correct.sum()/len(correct)
    return acc


class RNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_mat):
        """
        :param input_mat: a matrix [sentence length, batch size]
        :return:
        """
        embedded = self.embedding(input_mat)
        # [sentence length, batch size, embedding_dim]
        output, hidden = self.rnn(embedded)
        # output : [sentence length, batch size, hidden_dim]
        # hidden : [1, batch size, hidden_dim]
        return self.fc(hidden.squeeze(0))


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
    return epoch_loss / len(iterator), epoch_loss / len(iterator)


if __name__ == '__main__':
    # initialize
    SEED = 14
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # create dataset
    TEXT = data.Field()
    LABEL = data.LabelField(tensor_type=torch.FloatTensor)
    fields = [('text', TEXT), ('label', LABEL)]
    paths = {'train': 'train.csv', 'test': 'test.csv'}
    train, valid, test = dataset.get_dataset(fields=fields, paths=paths,
                                             view_info=True, valid_seed=SEED)
    # create vocabulary
    TEXT.build_vocab(train, max_size=2500)
    LABEL.build_vocab(train)
    print('TEXT vocabulary: {}\nlabel size {}'
          .format(len(TEXT.vocab), len(LABEL.vocab)))
    print(LABEL.vocab.stoi)
    print(TEXT.vocab.itos[:10])

    # create batch
    batch_size = 64
    train_iter, valid_iter, test_iter = data.BucketIterator.splits(
        (train, valid, test),
        batch_size=batch_size,
        sort_key=lambda x: len(x.text),
        repeat=False
    )

    # create network
    input_dim = len(TEXT.vocab)
    embedding_dim = 16
    hidden_dim = 32
    output_dim = 1
    model = RNN(input_dim, embedding_dim, hidden_dim, output_dim)

    # train the model
    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    model = model.to(device)
    criterion = criterion.to(device)
    n_epochs = 500
    for epoch in range(n_epochs):
        train_loss, train_acc = train_(model, train_iter, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_iter, criterion)
        print(
            f'epoch: {epoch+1:03}\ntrain loss: {train_loss:3f}, train acc: {train_acc*100:.2f}% '
            f'eval . loss: {valid_loss:.3f}, val . acc: {valid_acc*100:.2f}%')

    test_loss, test_acc = evaluate(model, test_iter, criterion)
    print(f'Test Loss: {test_loss:.3f}, Test Acc: {test_acc*100:.2f}%')