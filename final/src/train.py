import sys
import pickle
import math
import random
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
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--hidden_size', type=int, default=256)
arg = parser.parse_args()

max_seq = 16
input_size = 39
output_size = 13
data_size=2239

if __name__ == '__main__':
    print('Loading data.')
    id_train, x_train, y_train = load_data('data/train.dat', 'labels/Clean08TR_sp.mlf')
    id_test, x_test, y_test = load_data('data/test.dat', 'labels/answer.mlf')
    #pickle.dump(((id_train, x_train, y_train), (id_test, x_test, y_test)), open('data.pkl', 'wb')) 
    #(id_train, x_train, y_train), (id_test, x_test, y_test)= pickle.load(open('data.pkl', 'rb')) 

    train_loader = DataLoader(
        dataset=TensorDataset(torch.FloatTensor(x_train), torch.LongTensor(y_train)),
        batch_size=arg.batch_size,
        shuffle=True,
        num_workers=2
    )
    
    test_loader = DataLoader(
        dataset=TensorDataset(torch.FloatTensor(x_test), torch.LongTensor(y_test)),
        batch_size=arg.batch_size,
        shuffle=False,
        num_workers=2
    )

    print('Initial model.')

    encoder = Encoder(input_size, arg.hidden_size)
    decoder = Decoder(arg.hidden_size, output_size)
    enc_optim = torch.optim.RMSprop(encoder.parameters(), lr=arg.lr, alpha=0.9)
    dec_optim = torch.optim.RMSprop(decoder.parameters(), lr=arg.lr, alpha=0.9)
    citerion = torch.nn.CrossEntropyLoss()

    print('Start training')
    best_acc = 0
    for epoch in range(1, arg.epochs + 1):
        encoder.train()
        decoder.train()
        total_loss = 0
        eps = 0.5 if epoch <= 20 else 0.0
        for i_batch, (x_batch, y_batch) in enumerate(train_loader):
            l_batch = x_batch.size(0)
            x_batch = Variable(x_batch)
            y_batch = Variable(y_batch)
            
            enc_out, hidden = encoder(x_batch)

            output = Variable(torch.LongTensor(np.zeros((l_batch, 1), dtype=np.int64)))

            loss = 0
            use_teacher_forcing = True if random.random() < eps else False
            for t in range(max_seq):
                output, hidden = decoder(output, hidden, enc_out)
                loss += citerion(output.squeeze(), y_batch[:,t])
                output = output.max(-1)[1] if use_teacher_forcing else y_batch[:,t].unsqueeze(1)

            enc_optim.zero_grad()
            dec_optim.zero_grad()
            loss.backward()
            enc_optim.step()
            dec_optim.step()

            total_loss += loss.data[0]

        encoder.eval()
        decoder.eval()
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
                predict[-1].append(output.data.cpu().numpy())
            predict[-1] = np.hstack(predict[-1])
        predict = np.vstack(predict)
        
        ed, total = 0, 0
        sent_acc = []
        for pred, truth in zip(predict, y_test):
            pred = ''.join([itop[p] for p in pred]) 
            truth = ''.join([itop[p] for p in truth])
            sent_acc.append((pred == truth))
            ed += editdistance.eval(pred, truth)
            total += len(truth)
        
        print('Epoch: {}, train_loss: {:.4f}, eps: {:.4f}'.format(epoch, total_loss / len(train_loader), eps))
        print('Test: WORD={:.2f}, SENT={:.2f}'.format((1 - ed / total) * 100, np.mean(sent_acc) * 100))
        if ((1 - ed / total) * 100) > best_acc:
            best_acc = (1 - ed / total) * 100
            print('Save model.')
            with open('encoder.pt', 'wb') as file:
                torch.save(encoder.state_dict(), file)
            with open('decoder.pt', 'wb') as file:
                torch.save(decoder.state_dict(), file)
    

