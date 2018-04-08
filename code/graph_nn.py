""""""

import random

import torch
from torch.utils.data import TensorDataset, DataLoader


class GraphClassifier(torch.nn.Module):

    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super().__init__()
        self.input = torch.nn.Linear(input_size, hidden_size)
        self.activation = torch.nn.ReLU()
        self.hidden_layers = torch.nn.ModuleList(
            [torch.nn.Linear(hidden_size, hidden_size) for i in range(num_layers)])
        self.output = torch.nn.Linear(hidden_size, output_size)
        self.softmax = torch.nn.LogSoftmax(-1)

    def forward(self, x):
        hidden_state = self.activation(self.input(x))
        for layer in self.hidden_layers:
            hidden_state = self.activation(layer(hidden_state))
        output_state = self.output(hidden_state)
        output = self.softmax(output_state)
        return output


def convert_to_boolean(element):
    if element == "Y":
        element = 1
    else:
        element = 0
    return element


def get_graphs(size, file_suffix="", batch_size=64, num_workers=0):
    with open("graphs" + str(size) + file_suffix + ".txt") as graph_file:
        graph_lines = graph_file.read().splitlines()
    num_edges = size * (size-1) // 2
    graphs = [(line.split(" ")[:num_edges], line.split(" ")[num_edges]) 
              for line in graph_lines]
    graphs = [(torch.FloatTensor([int(value) for value in graph[0]]).unsqueeze(0), 
               torch.LongTensor([convert_to_boolean(graph[1])])) for graph in graphs]
    random.shuffle(graphs)
    train_graphs = graphs[:(len(graphs)*9)//10]
    eval_graphs = graphs[(len(graphs)*9)//10:]
    train_dataset = TensorDataset(torch.cat(list(zip(*train_graphs))[0]), 
                                  torch.cat(list(zip(*train_graphs))[1]))
    eval_dataset = TensorDataset(torch.cat(list(zip(*eval_graphs))[0]), 
                                 torch.cat(list(zip(*eval_graphs))[1]))
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                   shuffle=True, num_workers=num_workers)
    eval_data_loader = DataLoader(eval_dataset, batch_size=batch_size, 
                                  shuffle=True, num_workers=num_workers)
    return train_data_loader, eval_data_loader


def train_classifier(size, use_gpu=False):
    num_edges = size * (size-1) // 2
    classifier = GraphClassifier(num_edges, 100, 2, 2)
    if use_gpu:
        classifier = classifier.cuda()
    loss_function = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(classifier.parameters())
    train_data_loader, eval_data_loader = get_graphs(size)
    epochs = 50
    for i in range(epochs):
        for graphs, connected_indicators in train_data_loader:
            optimizer.zero_grad()
            target = torch.autograd.Variable(connected_indicators)
            input_ = torch.autograd.Variable(graphs)
            if use_gpu:
                input_ = input_.cuda()
                target = target.cuda()
            output = classifier(input_)
            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()

    return classifier, eval_data_loader


def test_classifier(classifier, size, eval_data_loader=None):
    classifier.cpu()
    if eval_data_loader is None:
        _, eval_data_loader = get_graphs(size)
    accuracy = 0
    for graphs, connected_indicators in eval_data_loader:
        output = classifier(torch.autograd.Variable(graphs))
        accuracy += output.max(1)[1].data.cpu().eq(
            connected_indicators).float().mean()

    return 100 * (accuracy / len(eval_data_loader))