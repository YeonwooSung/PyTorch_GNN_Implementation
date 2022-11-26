import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam

from graphsage import GraphSAGE
from cora_utils import CoraDataset


def main():
    hidden_dims = 16
    agg_class = 'max-pooling'
    dropout = 0.5
    num_layers = hidden_dims + 1
    batch_size = 8
    lr = 1e-3
    weight_decay = 5e-4
    print_every = 16
    epochs = 10000


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the data
    dataset = CoraDataset('./cora', 'train', num_layers, batch_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate_wrapper)
    
    input_dim, output_dim = dataset.get_dims()

    # Create the model
    model = GraphSAGE(input_dim, hidden_dims, output_dim, agg_class, dropout)
    model.train()

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    num_batches = int(math.ceil(len(dataset) / batch_size))

    print('--------------------------------')
    print('Training.')
    for epoch in range(epochs):
        print('Epoch {} / {}'.format(epoch+1, epochs))
        running_loss = 0.0
        num_correct, num_examples = 0, 0
        for (idx, batch) in enumerate(loader):
            features, node_layers, mappings, rows, labels = batch
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(features, node_layers, mappings, rows)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                running_loss += loss.item()
                predictions = torch.max(out, dim=1)[1]
                num_correct += torch.sum(predictions == labels).item()
                num_examples += len(labels)
            if (idx + 1) % print_every == 0:
                running_loss /= print_every
                accuracy = num_correct / num_examples
                print('    Batch {} / {}: loss {}, accuracy {}'.format(idx+1, num_batches, running_loss, accuracy))
                running_loss = 0.0
                num_correct, num_examples = 0, 0
        print('Finished training.')
        print('--------------------------------')


if __name__ == '__main__':
    main()
