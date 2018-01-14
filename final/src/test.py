import sys
import pickle
import numpy as np
from utils import *
from model import *
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import editdistance
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--hidden_size', type=int, default=256)
arg = parser.parse_args()

max_seq = 16
input_size = 39
output_size = 13

if __name__ == '__main__':
    print('Loading data.')
    #id_train, x_train, y_train = load_data('data/train.dat', 'labels/Clean08TR.mlf')
    id_test, x_test, y_test = load_data('data/test.dat', 'labels/answer.mlf')
    #pickle.dump(((id_train, x_train, y_train), (id_test, x_test, y_test)), open('data.pkl', 'wb')) 
    #(id_train, x_train, y_train), (id_test, x_test, y_test)= pickle.load(open('data.pkl', 'rb')) 
    test_loader = DataLoader(
        dataset=TensorDataset(torch.FloatTensor(x_test), torch.LongTensor(y_test)),
        batch_size=arg.batch_size,
        shuffle=False,
        num_workers=2
    )

    print('Initial model.')
    encoder = Encoder(input_size, arg.hidden_size)
    decoder = Decoder(arg.hidden_size, output_size)
    encoder.eval()
    decoder.eval()
    encoder.load_state_dict(torch.load('encoder.pt'))
    decoder.load_state_dict(torch.load('decoder.pt'))
    
    print('Predicting.')
    predict = []
    for x_batch, y_batch in test_loader:
        l_batch = x_batch.size(0)
        x_batch = Variable(x_batch)
        y_batch = Variable(y_batch)

        enc_out, hidden = encoder(x_batch)

        output = Variable(torch.LongTensor(np.zeros((l_batch, 1), dtype=np.int64)))
        
        predict.append([])
        for t in range(max_seq):
            output, hidden = decoder(output, hidden, enc_out)
            output = output.max(-1)[1]
            predict[-1].append(output.data.numpy())
        predict[-1] = np.hstack(predict[-1])
    predict = np.vstack(predict)
    
    print('Output.')
    ed, total = 0, 0
    sent_acc = []
    for pred, truth in zip(predict, y_test):
        pred = ''.join([itop[p] for p in pred]) 
        truth = ''.join([itop[p] for p in truth])
        sent_acc.append((pred == truth))
        ed += editdistance.eval(pred, truth)
        total += len(truth)
    print('SENT: {:.4f}%'.format(np.mean(sent_acc) * 100))
    print('WORD: {:.4f}%'.format((1 - ed / total) * 100))
        
