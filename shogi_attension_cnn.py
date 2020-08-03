import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dlshogi.common import *
from dlshogi import cppshogi

import argparse
import random
import os
import logging

parser = argparse.ArgumentParser()
parser.add_argument('train_data', type=str, nargs='+')
parser.add_argument('test_data', type=str)
parser.add_argument('--batchsize', '-b', type=int, default=1024)
parser.add_argument('--testbatchsize', type=int, default=640)
parser.add_argument('--epoch', '-e', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--weightdecay_rate', type=float, default=0.0001)
parser.add_argument('--beta', type=float, default=0.001)
parser.add_argument('--val_lambda', type=float, default=0.333)
parser.add_argument('--gpu', '-g', type=int, default=0)
parser.add_argument('--eval_interval', type=int, default=100)
args = parser.parse_args()

np.random.seed(0)
logging.basicConfig(format='%(asctime)s\t%(levelname)s\t%(message)s', datefmt='%Y/%m/%d %H:%M:%S', level=logging.DEBUG)

# 81マス + 持ち駒 × 2(先後)
HANDS_FEATURES_NUM = (8 + 4 + 4 + 4 + 4 + 2 + 2) * 2
L = 81 + HANDS_FEATURES_NUM
# 位置ごとの特徴ベクトルの次元 駒の種類と効きと利き数と位置(段と筋)、持ち駒の枚数表現
FEATURES_NUM = FEATURES1_NUM + 9 + 9 + HANDS_FEATURES_NUM
# 埋め込みベクトルの次元
E = 64

class Bias(nn.Module):
    def __init__(self, shape):
        super(Bias, self).__init__()
        self.bias=nn.Parameter(torch.Tensor(shape))

    def forward(self, input):
        return input + self.bias

k = 192
fcl = 256 # fully connected layers
class PolicyValueNetwork(nn.Module):
    def __init__(self, nhead=8, nlayers=1):
        super(PolicyValueNetwork, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer, Linear, BatchNorm1d
        self.embedding = Linear(FEATURES_NUM, E, bias=False)
        self.embedding_norm = BatchNorm1d(E)
        encoder_layers = TransformerEncoderLayer(E, nhead, E)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.l1_1 = nn.Conv2d(in_channels=E, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l1_2 = Linear(HANDS_FEATURES_NUM * E, k, bias=False)
        self.l2 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l3 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l4 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l5 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l6 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l7 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l8 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l9 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l10 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l11 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l12 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l13 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l14 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l15 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l16 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l17 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l18 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l19 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l20 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l21 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        # policy network
        self.l22 = nn.Conv2d(in_channels=k, out_channels=MAX_MOVE_LABEL_NUM, kernel_size=1, bias=False)
        self.l22_2 = Bias(9*9*MAX_MOVE_LABEL_NUM)
        # value network
        self.l22_v = nn.Conv2d(in_channels=k, out_channels=MAX_MOVE_LABEL_NUM, kernel_size=1)
        self.l23_v = nn.Linear(9*9*MAX_MOVE_LABEL_NUM, fcl)
        self.l24_v = nn.Linear(fcl, 1)

        #self.norm = nn.BatchNorm2d(E*2)
        self.norm1 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm2 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm3 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm4 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm5 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm6 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm7 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm8 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm9 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm10 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm11 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm12 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm13 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm14 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm15 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm16 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm17 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm18 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm19 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm20 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm21 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm22_v = nn.BatchNorm2d(MAX_MOVE_LABEL_NUM)

    def forward(self, src):
        h = self.embedding_norm(self.embedding(src.view(-1, FEATURES_NUM))).view(L, -1, E)
        h = self.transformer_encoder(h).permute(1, 2, 0)
        #h = torch.cat((h[:, :, :81], torch.mean(h[:, :, 81:], 2, keepdim=True).expand(-1, -1, 81)), 1).view(-1, E*2, 9, 9)
        #u1 = self.l1(self.norm(h))
        #u1 = self.l1(h)
        u1_1 = self.l1_1(h[:, :, :81].view(-1, E, 9, 9))
        u1_2 = self.l1_2(h[:, :, 81:].reshape(-1, HANDS_FEATURES_NUM * E)).unsqueeze(2).expand(-1, -1, 81).view(-1, k, 9, 9)
        u1 = u1_1 + u1_2
        # Residual block
        h1 = F.relu(self.norm1(u1))
        h2 = F.relu(self.norm2(self.l2(h1)))
        u3 = self.l3(h2) + u1
        # Residual block
        h3 = F.relu(self.norm3(u3))
        h4 = F.relu(self.norm4(self.l4(h3)))
        u5 = self.l5(h4) + u3
        # Residual block
        h5 = F.relu(self.norm5(u5))
        h6 = F.relu(self.norm6(self.l6(h5)))
        u7 = self.l7(h6) + u5
        # Residual block
        h7 = F.relu(self.norm7(u7))
        h8 = F.relu(self.norm8(self.l8(h7)))
        u9 = self.l9(h8) + u7
        # Residual block
        h9 = F.relu(self.norm9(u9))
        h10 = F.relu(self.norm10(self.l10(h9)))
        u11 = self.l11(h10) + u9
        # Residual block
        h11 = F.relu(self.norm11(u11))
        h12 = F.relu(self.norm12(self.l12(h11)))
        u13 = self.l13(h12) + u11
        # Residual block
        h13 = F.relu(self.norm13(u13))
        h14 = F.relu(self.norm14(self.l14(h13)))
        u15 = self.l15(h14) + u13
        # Residual block
        h15 = F.relu(self.norm15(u15))
        h16 = F.relu(self.norm16(self.l16(h15)))
        u17 = self.l17(h16) + u15
        # Residual block
        h17 = F.relu(self.norm17(u17))
        h18 = F.relu(self.norm18(self.l18(h17)))
        u19 = self.l19(h18) + u17
        # Residual block
        h19 = F.relu(self.norm19(u19))
        h20 = F.relu(self.norm20(self.l20(h19)))
        u21 = self.l21(h20) + u19
        h21 = F.relu(self.norm21(u21))
        # policy network
        h22 = self.l22(h21)
        h22_1 = self.l22_2(h22.view(-1, 9*9*MAX_MOVE_LABEL_NUM))
        # value network
        h22_v = F.relu(self.norm22_v(self.l22_v(h21)))
        h23_v = F.relu(self.l23_v(h22_v.view(-1, 9*9*MAX_MOVE_LABEL_NUM)))

        return h22_1, self.l24_v(h23_v)

torch.cuda.set_device(args.gpu)
device = torch.device("cuda")

model = PolicyValueNetwork()
model.to(device)

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weightdecay_rate, nesterov=True)
cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
bce_with_logits_loss = torch.nn.BCEWithLogitsLoss()

logging.debug('read teacher data')
def load_teacher(files):
    data = []
    for path in files:
        if os.path.exists(path):
            logging.debug(path)
            data.append(np.fromfile(path, dtype=HuffmanCodedPosAndEval))
        else:
            logging.debug('{} not found, skipping'.format(path))
    return np.concatenate(data)
train_data = load_teacher(args.train_data)
logging.debug('read test data')
logging.debug(args.test_data)
test_data = np.fromfile(args.test_data, dtype=HuffmanCodedPosAndEval)

logging.info('train position num = {}'.format(len(train_data)))
logging.info('test position num = {}'.format(len(test_data)))


# mini batch
def mini_batch(hcpevec):
    features1 = np.empty((len(hcpevec), FEATURES1_NUM, 9, 9), dtype=np.float32)
    features2 = np.empty((len(hcpevec), FEATURES2_NUM, 9, 9), dtype=np.float32)
    features = np.zeros((L, len(hcpevec), FEATURES_NUM), dtype=np.float32)
    move = np.empty((len(hcpevec)), dtype=np.int32)
    result = np.empty((len(hcpevec)), dtype=np.float32)
    value = np.empty((len(hcpevec)), dtype=np.float32)

    cppshogi.hcpe_decode_with_value(hcpevec, features1, features2, move, result, value)

    # convert features
    # 盤上の駒
    features1_t = features1.transpose(2, 3, 0, 1).reshape(81, -1, FEATURES1_NUM)
    features[:81,:,:FEATURES1_NUM] = features1_t

    # 位置
    # 段、筋をワンホットベクトルで表す
    features[:81,:,FEATURES1_NUM:FEATURES1_NUM + 9 + 9] = np.c_[np.tile(np.eye(9, dtype=np.float32), (1, 9)).reshape((81,9)), np.tile(np.eye(9, dtype=np.float32), (9, 1))].reshape(81, -1, 9 + 9)

    # 持ち駒
    features2_t = features2.transpose(1, 0, 2, 3).reshape(FEATURES2_NUM, -1, 81)
    #features[81:,:,FEATURES1_NUM:FEATURES1_NUM + 9 + 9] = features2_t[:56,:,:18]
    features[81:,:,FEATURES1_NUM + 9 + 9:] = features2_t[:56,:,:56] * np.eye(56).reshape((56, 1, 56))

    z = result.astype(np.float32) - value + 0.5

    return (torch.tensor(features).to(device),
            torch.tensor(move.astype(np.int64)).to(device),
            torch.tensor(result.reshape((len(hcpevec), 1))).to(device),
            torch.tensor(z).to(device),
            torch.tensor(value.reshape((len(value), 1))).to(device)
            )

def accuracy(y, t):
    return (torch.max(y, 1)[1] == t).sum().item() / len(t)

def binary_accuracy(y, t):
    pred = y >= 0
    truth = t >= 0.5
    return pred.eq(truth).sum().item() / len(t)

# train
t = 0
itr = 0
sum_loss1 = 0
sum_loss2 = 0
sum_loss3 = 0
sum_loss = 0
eval_interval = args.eval_interval
for e in range(args.epoch):
    np.random.shuffle(train_data)

    itr_epoch = 0
    sum_loss1_epoch = 0
    sum_loss2_epoch = 0
    sum_loss3_epoch = 0
    sum_loss_epoch = 0
    for i in range(0, len(train_data) - args.batchsize + 1, args.batchsize):
        model.train()

        x, t1, t2, z, value = mini_batch(train_data[i:i+args.batchsize])
        y1, y2 = model(x)

        model.zero_grad()
        loss1 = (cross_entropy_loss(y1, t1) * z).mean()
        if args.beta > 0:
            loss1 += args.beta * (F.softmax(y1, dim=1) * F.log_softmax(y1, dim=1)).sum(dim=1).mean()
        loss2 = bce_with_logits_loss(y2, t2)
        loss3 = bce_with_logits_loss(y2, value)
        loss = loss1 + (1 - args.val_lambda) * loss2 + args.val_lambda * loss3
        loss.backward()
        optimizer.step()

        t += 1
        itr += 1
        sum_loss1 += loss1.item()
        sum_loss2 += loss2.item()
        sum_loss3 += loss3.item()
        sum_loss += loss.item()
        itr_epoch += 1
        sum_loss1_epoch += loss1.item()
        sum_loss2_epoch += loss2.item()
        sum_loss3_epoch += loss3.item()
        sum_loss_epoch += loss.item()

        # print train loss
        if t % eval_interval == 0:
            model.eval()

            x, t1, t2, z, value = mini_batch(np.random.choice(test_data, args.testbatchsize))
            with torch.no_grad():
                y1, y2 = model(x)

                loss1 = (cross_entropy_loss(y1, t1) * z).mean()
                loss2 = bce_with_logits_loss(y2, t2)
                loss3 = bce_with_logits_loss(y2, value)
                loss = loss1 + (1 - args.val_lambda) * loss2 + args.val_lambda * loss3

                logging.info('epoch = {}, iteration = {}, loss = {:.8}, {:.8}, {:.8}, {:.8}, test loss = {:.8}, {:.8}, {:.8}, {:.8}, test accuracy = {:.8}, {:.8}'.format(
                    e + 1, t,
                    sum_loss1 / itr, sum_loss2 / itr, sum_loss3 / itr, sum_loss / itr,
                    loss1.item(), loss2.item(), loss3.item(), loss.item(),
                    accuracy(y1, t1), binary_accuracy(y2, t2)))
            itr = 0
            sum_loss1 = 0
            sum_loss2 = 0
            sum_loss3 = 0
            sum_loss = 0

    # print train loss for each epoch
    itr_test = 0
    sum_test_loss1 = 0
    sum_test_loss2 = 0
    sum_test_loss3 = 0
    sum_test_loss = 0
    sum_test_accuracy1 = 0
    sum_test_accuracy2 = 0
    sum_test_entropy1 = 0
    sum_test_entropy2 = 0
    model.eval()
    with torch.no_grad():
        for i in range(0, len(test_data) - args.testbatchsize, args.testbatchsize):
            x, t1, t2, z, value = mini_batch(test_data[i:i+args.testbatchsize])
            y1, y2 = model(x)

            itr_test += 1
            loss1 = (cross_entropy_loss(y1, t1) * z).mean()
            loss2 = bce_with_logits_loss(y2, t2)
            loss3 = bce_with_logits_loss(y2, value)
            loss = loss1 + (1 - args.val_lambda) * loss2 + args.val_lambda * loss3
            sum_test_loss1 += loss1.item()
            sum_test_loss2 += loss2.item()
            sum_test_loss3 += loss3.item()
            sum_test_loss += loss.item()
            sum_test_accuracy1 += accuracy(y1, t1)
            sum_test_accuracy2 += binary_accuracy(y2, t2)

            entropy1 = (- F.softmax(y1, dim=1) * F.log_softmax(y1, dim=1)).sum(dim=1)
            sum_test_entropy1 += entropy1.mean().item()

            p2 = y2.sigmoid()
            #entropy2 = -(p2 * F.log(p2) + (1 - p2) * F.log(1 - p2))
            log1p_ey2 = F.softplus(y2)
            entropy2 = -(p2 * (y2 - log1p_ey2) + (1 - p2) * -log1p_ey2)
            sum_test_entropy2 +=entropy2.mean().item()

        logging.info('epoch = {}, iteration = {}, train loss avr = {:.8}, {:.8}, {:.8}, {:.8}, test_loss = {:.8}, {:.8}, {:.8}, {:.8}, test accuracy = {:.8}, {:.8}, test entropy = {:.8}, {:.8}'.format(
            e + 1, t,
            sum_loss1_epoch / itr_epoch, sum_loss2_epoch / itr_epoch, sum_loss3_epoch / itr_epoch, sum_loss_epoch / itr_epoch,
            sum_test_loss1 / itr_test, sum_test_loss2 / itr_test, sum_test_loss3 / itr_test, sum_test_loss / itr_test,
            sum_test_accuracy1 / itr_test, sum_test_accuracy2 / itr_test,
            sum_test_entropy1 / itr_test, sum_test_entropy2 / itr_test))
