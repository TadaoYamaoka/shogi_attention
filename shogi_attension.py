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
L = 81 + (8 + 4 + 4 + 4 + 4 + 2 + 2) * 2
# 位置ごとの特徴ベクトルの次元 駒の種類と効きと利き数と位置(段と筋)、持ち駒の枚数表現
FEATURES_NUM = FEATURES1_NUM + 9 + 9 + (8 + 4 + 4 + 4 + 4 + 2 + 2) * 2
# 埋め込みベクトルの次元
E = 64
# 移動を表すラベルの数 座標×移動方向
MOVE_LABEL_NUM = 81 * 27

class PolicyValueNetwork(nn.Module):
    def __init__(self, nhead=8, nlayers=1):
        super(PolicyValueNetwork, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer, Linear, BatchNorm1d
        self.embedding = Linear(FEATURES_NUM, E, bias=False)
        self.embedding_norm = BatchNorm1d(E)
        encoder_layers = TransformerEncoderLayer(E, nhead, E)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.norm = BatchNorm1d(L * E)
        self.policy_fcl1 = Linear(L * E, 256)
        self.policy_fcl2 = Linear(256, MOVE_LABEL_NUM)
        self.value_fcl1 = Linear(L * E, 256)
        self.value_fcl2 = Linear(256, 1)

    def forward(self, src):
        h = self.embedding_norm(self.embedding(src.view(-1, FEATURES_NUM))).view(L, -1, E)
        h = self.transformer_encoder(h).permute(1, 0, 2)
        h = self.norm(h.reshape(-1, L * E))
        # policy
        h_p = F.relu(self.policy_fcl1(h))
        h_p = self.policy_fcl2(h_p)
        # value
        h_v = F.relu(self.value_fcl1(h))
        h_v = self.value_fcl2(h_v)

        return h_p, h_v

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
