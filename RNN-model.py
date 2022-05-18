import numpy as np
import random
import sys
import csv
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torchinfo import summary
from transformer_encoder import return_model_data
#忽略警告
import warnings
warnings.filterwarnings("ignore")

num_iter = 10

past_chunk = 0
future_chunk = 1
hidden_size = 32
num_layers = 1

# only one can be set 1
use_embedding = 1
use_linear_reduction = 0
###
atten_decoder = 1
use_dropout = 0
use_average_embedding = 1

weight = 10
labmda = 0
topk_labels = 3

# It should be the same as the reductioned input in decoder's cat function

teacher_forcing_ratio = 0
MAX_LENGTH = 1000
learning_rate = 0.001
optimizer_option = 2
print_val = 3000
use_cuda = torch.cuda.is_available()


class EncoderRNN_new(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(EncoderRNN_new, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reduction = nn.Linear(input_size, hidden_size)
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.time_embedding = nn.Embedding(input_size, hidden_size)
        self.time_weight = nn.Linear(input_size, input_size)
        if use_embedding or use_linear_reduction:
            self.gru = nn.GRU(hidden_size, hidden_size, num_layers)
        else:
            self.gru = nn.GRU(input_size, hidden_size, num_layers)

    def forward(self, input, hidden):
        if use_embedding:
            # print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
            list = Variable(torch.LongTensor(input).view(-1, 1))
            if use_cuda:
                list = list
            average_embedding = Variable(torch.zeros(hidden_size)).view(1, 1, -1)
            # sum_embedding = Variable(torch.zeros(hidden_size)).view(1,1,-1)
            vectorized_input = Variable(torch.zeros(self.input_size)).view(-1)#没用到
            if use_cuda:
                average_embedding = average_embedding
                # sum_embedding = sum_embedding.cuda()
                vectorized_input = vectorized_input
            cont = 0
            for ele in list:    #此for循环将一个诊疗日中所出现的所有诊疗活动的embedding相加
                embedded = self.embedding(ele).view(1, 1, -1)
                tmp = average_embedding.clone() #Tensor.clone()得到的变量与原变量不共享内存空间
                average_embedding = tmp + embedded
                # print("embedded:", embedded)
                # print("average_embedding:", average_embedding)
                # embedded = self.time_embedding(ele).view(1, 1, -1)
                # tmp = sum_embedding.clone()
                # sum_embedding = tmp + embedded
                vectorized_input[ele] = 1
                cont += 1

            # normalize_length = Variable(torch.LongTensor(len(idx_list)))
            # if use_cuda:
            #     normalize_length = normalize_length.cuda()
            if use_average_embedding:
                tmp = [cont] * hidden_size
                length = Variable(torch.FloatTensor(tmp))
                if use_cuda:
                    length = length.cuda()
                # for idx in range(hidden_size):
                real_ave = average_embedding.view(-1) / length
                average_embedding = real_ave.view(1, 1, -1)

            embedding = average_embedding
        else:
            tensorized_input = torch.from_numpy(input).clone().type(torch.FloatTensor)
            inputs = Variable(torch.unsqueeze(tensorized_input, 0).view(1, -1))
            if use_cuda:
                inputs = inputs.cuda()
            if use_linear_reduction == 1:
                reduced_input = self.reduction(inputs)
            else:
                reduced_input = inputs

            embedding = torch.unsqueeze(reduced_input, 0)
        # print("embedding:", embedding)
        output, hidden = self.gru(embedding, hidden)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(num_layers, 1, self.hidden_size))
        if use_cuda:
            # 返回一个1*1*32的tensor
            return result.cuda()
        else:
            return result


#


class AttnDecoderRNN_new(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, dropout_p=0.2, max_length=MAX_LENGTH):
        super(AttnDecoderRNN_new, self).__init__()
        self.hidden_size = hidden_size      #32
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        if use_embedding or use_linear_reduction:
            self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
            self.attn1 = nn.Linear(self.hidden_size + output_size, self.hidden_size)    
        else:
            self.attn = nn.Linear(self.hidden_size + self.output_size, self.output_size)

        if use_embedding or use_linear_reduction:
            self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
            #attn_combine3后面没用到
            self.attn_combine3 = nn.Linear(self.hidden_size * 2 + output_size, self.hidden_size)
        else:
            self.attn_combine = nn.Linear(self.hidden_size + self.output_size, self.hidden_size)
        self.attn_combine5 = nn.Linear(self.output_size, self.output_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.reduction = nn.Linear(self.output_size, self.hidden_size)
        if use_embedding or use_linear_reduction:
            self.gru = nn.GRU(hidden_size, hidden_size, num_layers)
        else:
            self.gru = nn.GRU(hidden_size, hidden_size, num_layers)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs, history_record, last_hidden):
        if use_embedding:
            list = Variable(torch.LongTensor(input).view(-1, 1))
            if use_cuda:
                list = list.cuda()
            average_embedding = Variable(torch.zeros(hidden_size)).view(1, 1, -1)
            if use_cuda:
                average_embedding = average_embedding.cuda()

            cont = 0
            for ele in list:
                embedded = self.embedding(ele).view(1, 1, -1)
                tmp = average_embedding.clone()
                average_embedding = tmp + embedded
                cont += 1

            if use_average_embedding:
                tmp = [cont] * hidden_size
                length = Variable(torch.FloatTensor(tmp))
                if use_cuda:
                    length = length.cuda()
                # for idx in range(hidden_size):
                real_ave = average_embedding.view(-1) / length
                average_embedding = real_ave.view(1, 1, -1)

            embedding = average_embedding
        else:
            tensorized_input = torch.from_numpy(input).clone().type(torch.FloatTensor)
            inputs = Variable(torch.unsqueeze(tensorized_input, 0).view(1, -1))
            if use_cuda:
                inputs = inputs.cuda()
            if use_linear_reduction == 1:
                reduced_input = self.reduction(inputs)
            else:
                reduced_input = inputs

            embedding = torch.unsqueeze(reduced_input, 0)

        if use_dropout:
            droped_ave_embedded = self.dropout(embedding)#相当于vkdroped_ave_embedded即经过平均池化和dropout的输入embedding
        else:
            droped_ave_embedded = embedding

        history_context = Variable(torch.FloatTensor(history_record).view(1, -1))   #γ向量
        if use_cuda:
            history_context = history_context.cuda()

        attn_weights = F.softmax(
            self.attn(torch.cat((droped_ave_embedded[0], hidden[0]), 1)), dim=1)
        # print("*"*50)
        # print("droped_ave_embedded shape:", droped_ave_embedded.shape)
        # print("hidden shape:", hidden.shape)
        # print("attn_weights shape:", attn_weights.shape)
        # print("encoder_outputs shape:", encoder_outputs.shape)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))
        # print("attn_applied shape:", attn_applied.shape)

        # element_attn_weights = F.softmax(
        #     self.attn1(torch.cat((history_context, hidden[0]), 1)), dim=1)

        # attn_applied = torch.bmm(element_attn_weights.unsqueeze(0),encoder_outputs.unsqueeze(0))

        # attn_embedd = element_attn_weights * droped_ave_embedded[0]

        output = torch.cat((droped_ave_embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
        # output = torch.cat((droped_ave_embedded[0], attn_applied[0], time_coef.unsqueeze(0)), 1)
        # output = self.attn_combine3(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        linear_output = self.out(output[0]) #linear_output与one-hot向量维度一致
        # output_user_item = F.softmax(linear_output)

        value = torch.sigmoid(self.attn_combine5(history_context).unsqueeze(0)) #value即为论文中的α

        one_vec = Variable(torch.ones(self.output_size).view(1, -1))
        if use_cuda:
            one_vec = one_vec.cuda()

        # ones_set = torch.index_select(value[0,0], 1, ones_idx_set[:, 1])
        res = history_context.clone()
        res[history_context != 0] = 1

        output = F.softmax(linear_output * (one_vec - res * value[0]) + history_context * value[0], dim=1)

        return output.view(1, -1), hidden, attn_weights

    def initHidden(self):
        result = Variable(torch.zeros(num_layers, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


class custom_MultiLabelLoss_torch(nn.modules.loss._Loss):
    def __init__(self):
        super(custom_MultiLabelLoss_torch, self).__init__()

    def forward(self, pred, target, weights):
        mseloss = torch.sum(weights * torch.pow((pred - target), 2))
        pred = pred.data
        target = target.data
        #
        ones_idx_set = (target == 1).nonzero()
        zeros_idx_set = (target == 0).nonzero()
        # zeros_idx_set = (target == -1).nonzero()
        
        ones_set = torch.index_select(pred, 1, ones_idx_set[:, 1])
        zeros_set = torch.index_select(pred, 1, zeros_idx_set[:, 1])
        
        repeat_ones = ones_set.repeat(1, zeros_set.shape[1])
        repeat_zeros_set = torch.transpose(zeros_set.repeat(ones_set.shape[1], 1), 0, 1).clone()
        repeat_zeros = repeat_zeros_set.reshape(1, -1)
        difference_val = -(repeat_ones - repeat_zeros)
        exp_val = torch.exp(difference_val)
        exp_loss = torch.sum(exp_val)
        normalized_loss = exp_loss / (zeros_set.shape[1] * ones_set.shape[1])
        set_loss = Variable(torch.FloatTensor([labmda * normalized_loss]), requires_grad=True)
        if use_cuda:
            set_loss = set_loss.cuda()
        loss = mseloss + set_loss
        #loss = mseloss
        return loss


def generate_dictionary_BA(path, files, attributes_list):
    # path = '../Minnemudac/'
    # files = ['Coborn_history_order.csv','Coborn_future_order.csv']
    # files = ['BA_history_order.csv', 'BA_future_order.csv']
    # attributes_list = ['MATERIAL_NUMBER']
    dictionary_table = {}
    counter_table = {}
    for attr in attributes_list:
        dictionary = {}
        dictionary_table[attr] = dictionary
        counter_table[attr] = 0

    #####################################
    maxInt = sys.maxsize
    decrement = True

    while decrement:
    # decrease the maxInt value by factor 10 
    # as long as the OverflowError occurs.
 
        decrement = False
    try:
        csv.field_size_limit(maxInt)
    except OverflowError:
        maxInt = int(maxInt/10)
        decrement = True
    ######################################

    # csv.field_size_limit(sys.maxsize)
    for filename in files:
        # print("filename:",filename)
        count = 0
        with open(path + filename, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in reader:
                if count == 0:
                    count += 1
                    continue
                key = attributes_list[0]
                if row[2] not in dictionary_table[key]:
                    dictionary_table[key][row[2]] = counter_table[key]
                    counter_table[key] = counter_table[key] + 1
                    count += 1
        print("count:", count)

    print("counter_table:", counter_table)
    # print("dictionary_table:", dictionary_table)      #dictionary_table:把item的编号由字符串格式转成数字
    print("count:", count)

    total = 0
    for key in counter_table.keys():
        total = total + counter_table[key]

    print('# dimensions of final vector: ' + str(total) + ' | ' + str(count - 1))   #共5000个item，每个set设置one-hot向量，5000维

    return dictionary_table, total, counter_table       #dictionary_table:item编号字符串格式与数字格式对应， one-hot向量维度， 数据列名和向量维度对应


def read_claim2vector_embedding_file_no_vector(path, files):
    # attributes_list = ['DRG', 'PROVCAT ', 'RVNU_CD', 'DIAG', 'PROC']
    attributes_list = ['MATERIAL_NUMBER']       #数据集中项目列表名，每个set里包含的项目展开生成的列表
    # path = '../Minnemudac/'
    print('start dictionary generation...')
    # item编号        向量维度   数据列名对应向量维度
    dictionary_table, num_dim, counter_table = generate_dictionary_BA(path, files, attributes_list)
    print('finish dictionary generation*****')
    usr_attr = 'CUSTOMER_ID'
    ord_attr = 'ORDER_NUMBER'

    # dictionary_table, num_dim, counter_table = GDF.generate_dictionary(attributes_list)

    freq_max = 200
    ## all the follow three ways array. First index is patient, second index is the time step, third is the feature vector
    data_chunk = []
    day_gap_counter = []
    claims_counter = 0
    num_claim = 0
    code_freq_at_first_claim = np.zeros(num_dim + 2)

    for file_id in range(len(files)):

        count = 0
        data_chunk.append({})
        filename = files[file_id]
        with open(path + filename, 'r') as csvfile:
            # gap_within_one_year = np.zeros(365)
            reader = csv.DictReader(csvfile)
            last_pid_date = '*'
            last_pid = '-1'
            last_days = -1
            # 2 more elements in the end for start and end states
            feature_vector = []
            for row in reader:
                cur_pid_date = row[usr_attr] + '_' + row[ord_attr]      #cur_pid_date: 00046855  _1 cur_pid: 00046855 
                cur_pid = row[usr_attr]

                if cur_pid != last_pid:
                    # start state
                    tmp = [-1]
                    data_chunk[file_id][cur_pid] = []
                    data_chunk[file_id][cur_pid].append(tmp)
                    num_claim = 0

                if cur_pid_date not in last_pid_date:
                    if last_pid_date not in '*' and last_pid not in '-1':
                        sorted_feature_vector = np.sort(feature_vector)
                        data_chunk[file_id][last_pid].append(sorted_feature_vector)
                        if len(sorted_feature_vector) > 0:
                            count = count + 1
                        # data_chunk[file_id][last_pid].append(feature_vector)
                    feature_vector = []

                    claims_counter = 0
                if cur_pid != last_pid:
                    # end state
                    if last_pid not in '-1':
                        tmp = [-1]
                        data_chunk[file_id][last_pid].append(tmp)

                key = attributes_list[0]

                within_idx = dictionary_table[key][row[key]]
                previous_idx = 0

                for j in range(attributes_list.index(key)):
                    previous_idx = previous_idx + counter_table[attributes_list[j]]
                idx = within_idx + previous_idx

                # set corresponding dimention to 1
                if idx not in feature_vector:
                    feature_vector.append(idx)

                last_pid_date = cur_pid_date
                last_pid = cur_pid
                # last_days = cur_days
                if file_id == 1:
                    claims_counter = claims_counter + 1
                # print("feature_vector:", feature_vector)      每个order中的item排序
            if last_pid_date not in '*' and last_pid not in '-1':
                data_chunk[file_id][last_pid].append(np.sort(feature_vector))
        print('num of vectors having entries more than 1: ' + str(count))
    # print("data_chunk:", data_chunk)
    return data_chunk, num_dim + 2, code_freq_at_first_claim


def train(txt_input, txt_encoder, txt_encoder_optimizer, input_variable, target_variable, encoder, decoder, codes_inverse_freq, encoder_optimizer, decoder_optimizer,
          criterion, output_size, next_k_step, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()   #初始隐层向量   [1,1,32]

    encoder_optimizer.zero_grad()           #优化器梯度置0
    decoder_optimizer.zero_grad()
    txt_encoder_optimizer.zero_grad()

    input_length = len(input_variable)      #诊疗日个数
    target_length = len(target_variable)

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    if use_cuda:
        encoder_outputs = encoder_outputs.cuda()

    loss = 0

    history_record = np.zeros(output_size)
    for ei in range(input_length - 1):
        if ei == 0:
            continue
        for ele in input_variable[ei]:
            history_record[ele] += 1 / (input_length - 2)
    #文本处理模型
    # txt_input = torch.LongTensor(list(txt_input)).cuda()
    # txt_outputs, txt_self_attns = txt_encoder(txt_input)
    # txt_outputs = txt_outputs.squeeze()
    # txt_outputs = txt_outputs.mean(dim=0)
    # txt_outputs = txt_outputs.unsqueeze(0)
    # txt_outputs = txt_outputs.unsqueeze(0)
    # # print("txt_outputs shape:", txt_outputs.shape)
    # encoder_hidden = txt_outputs
    

    for ei in range(input_length - 1):
        if ei == 0:
            continue                      #input_variable[ei]为一个诊疗日所包含的诊疗项目数组
        encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)    #执行一次相当于经历一个rnn单元
        encoder_outputs[ei - 1] = encoder_output[0][0]      #encoder_outputs记录所有输出结果
        # print("##########################################")
        # print("encoder_output:", encoder_output)
        # print("encoder_outputs:", encoder_outputs)

    last_input = input_variable[input_length - 2]
    decoder_hidden = encoder_hidden
    last_hidden = encoder_hidden
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    num_str = 0
    topk = 1
    max_len = 5

    if next_k_step > 0:
        if next_k_step <= target_length - 2:
            max_step = next_k_step
        else:
            max_step = target_length - 2
    else:
        max_step = target_length - 1
        max_step = min(target_length - 2, max_len)
    decoder_input = last_input

    # print("++++++++++++++++++++++++++++++++++")
    # print("max_step:", max_step)
    for di in range(max_step):      #max_step大多数为1

        if atten_decoder:
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs, history_record, last_hidden)
        else:
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
        topv, topi = decoder_output.data.topk(topk) #返回所发现的第一个最大值及其下标
        ni = topi[0][0]
        # print("++++++++++++++++++++++++++++++++++")
        # print('decoder_output:', decoder_output, decoder_output.size())
        # print("topv:", topv)
        # print("topi:", topi)
        # print("ni:", ni)

        # activation_bound
        # topk_labels
        # target_neg = zero2neg(target_variable[di])

        vectorized_target = np.zeros(output_size)
        for idx in target_variable[di + 1]:
            vectorized_target[idx] = 1

        target = Variable(torch.FloatTensor(vectorized_target).view(1, -1))
        if use_cuda:
            target = target.cuda()
        weights = Variable(torch.FloatTensor(codes_inverse_freq).view(1, -1))
        if use_cuda:
            weights = weights.cuda()

        tt = criterion(decoder_output, target, weights)
        # tt = torch.sum(weights*torch.pow((decoder_output - target),2))
        loss += tt

        decoder_input = target_variable[di + 1]
        # loss += multilable_loss(decoder_output, target)

    # encoder_optimizer.zero_grad()
    # decoder_optimizer.zero_grad()
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()
    txt_encoder_optimizer.step()

    return loss.item() / target_length


######################################################################
# This is a helper function to print time elapsed and estimated time
# remaining given the current time and progress %.
#

import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def trainIters(txt_inputs, txt_encoder, data_chunk, output_size, encoder, decoder, model_id, training_key_set, codes_inverse_freq, next_k_step,
               n_iters, print_every=300):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    encoder_pathes = []
    decoder_pathes = []
    txt_encoder_pathes = []
    # elem_wise_connection.initWeight()

    # sum_history = add_history(data_chunk[past_chunk],training_key_set,output_size)
    # KNN_history = KNN_history_record1(sum_history, output_size, num_nearest_neighbors)
    KNN_history = []

    #优化器选择
    if optimizer_option == 1:
        encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
        txt_encoder_optimizer = optim.SGD(txt_encoder.parameters(), lr=learning_rate)
    elif optimizer_option == 2:
        # encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-09, weight_decay=0)
        # encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate, betas=(0.88, 0.95), eps=1e-08, weight_decay=0)
        encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-11,
                                             weight_decay=0)
        decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-11,
                                             weight_decay=0)
        txt_encoder_optimizer = torch.optim.Adam(txt_encoder.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-11,
                                             weight_decay=0)
    elif optimizer_option == 3:
        encoder_optimizer = torch.optim.RMSprop(encoder.parameters(), lr=learning_rate, alpha=0.99, eps=1e-08,
                                                weight_decay=0, momentum=0, centered=False)
        decoder_optimizer = torch.optim.RMSprop(decoder.parameters(), lr=learning_rate, alpha=0.99, eps=1e-08,
                                                weight_decay=0, momentum=0, centered=False)
        txt_encoder_optimizer = torch.optim.RMSprop(txt_encoder.parameters(), lr=learning_rate, alpha=0.99, eps=1e-08,
                                                weight_decay=0, momentum=0, centered=False)
    elif optimizer_option == 4:
        encoder_optimizer = torch.optim.Adadelta(encoder.parameters(), lr=learning_rate, rho=0.9, eps=1e-06,
                                                 weight_decay=0)
        decoder_optimizer = torch.optim.Adadelta(decoder.parameters(), lr=learning_rate, rho=0.9, eps=1e-06,
                                                 weight_decay=0)
        txt_encoder_optimizer = torch.optim.Adadelta(txt_encoder.parameters(), lr=learning_rate, rho=0.9, eps=1e-06,
                                                 weight_decay=0)

    # training_pairs = [variablesFromPair(random.choice(pairs))
    #                  for i in range(n_iters)]
    # criterion = nn.NLLLoss()
    total_iter = 0
    criterion = custom_MultiLabelLoss_torch()
    for j in range(n_iters):    #n_iters:20
        print("***********************************************")
        print(j)

        key_idx = np.random.permutation(len(training_key_set))  #生成一个训练集住院号个数长度的array，里面元素为序号，然后随机排序，permutation是随机排序
        # key_idx = np.random.choice(len(training_key_set),n_iters)
        training_keys = []

        for idx in key_idx:
            training_keys.append(training_key_set[idx])         #重新排序后的住院号序号

            # criterion = custom_MultiLabelLoss_MSE()
        # criterion = nn.MultiLabelSoftMarginLoss(size_average=False)
        # criterion = nn.BCELoss()
        weight_vector = []

        for iter in range(1, len(training_key_set) + 1):
            # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            # print(iter, "length training_key_set:", len(training_key_set))
            # training_pair = training_pairs[iter - 1]
            # input_variable = training_pair[0]
            # target_variable = training_pair[1]
            input_variable = data_chunk[past_chunk][training_keys[iter - 1]]    
            target_variable = data_chunk[future_chunk][training_keys[iter - 1]]
            # print("*"*40)
            tmp_no = training_keys[iter-1][4:]
            txt_input = txt_inputs[tmp_no]
            # print("txt_inputs:", txt_inputs[tmp_no])
            '''
            input_variable:[[-1], array([  0,   1,   3,   4,   6,   7,  10,  12,  14,  15,  17,  37,  39,
        42,  73,  74,  77, 124, 193]), array([  0,   1, 193]),[-1]]
            target_variable:[[-1], array([  0,   1,  22,  79, 193, 214]), array([  0,   1,  22,  79, 193, 214]),[-1]]
            '''

            loss = train(txt_input, txt_encoder, txt_encoder_optimizer, input_variable, target_variable, encoder,
                         decoder, codes_inverse_freq, encoder_optimizer, decoder_optimizer, criterion, output_size,
                         next_k_step)
            print_loss_total += loss
            plot_loss_total += loss
            # print('loss:', loss)

            total_iter += 1

        print_loss_avg = print_loss_total / len(training_key_set)
        print_loss_total = 0
        print('%s (%d %d%%) %.6f' % (timeSince(start, total_iter / (n_iters * len(training_key_set))), total_iter, total_iter / (n_iters * len(training_key_set)) * 100, print_loss_avg))

        filepath = './models/encoder' + (model_id) + '_model_epoch' + str(int(j))
        encoder_pathes.append(filepath)
        torch.save(encoder, filepath)

        filepath = './models/decoder' + (model_id) + '_model_epoch' + str(int(j))
        decoder_pathes.append(filepath)
        torch.save(decoder, filepath)

        filepath = './models/txt_encoder' + (model_id) + '_model_epoch' + str(int(j))
        txt_encoder_pathes.append(filepath)
        torch.save(txt_encoder, filepath)

        print('Finish epoch: ' + str(j))
        print('Model is saved.')
        sys.stdout.flush()
    # showPlot(plot_losses)
    # print('The loss: ' + str(print_loss_total))


######################################################################
# Plotting results
# ----------------
#
# Plotting is done with matplotlib, using the array of loss values
# ``plot_losses`` saved while training.
#

cosine_sim = []
pair_cosine_sim = []


def decoding_next_k_step(txt_input, txt_encoder, encoder, decoder, input_variable, target_variable, output_size, k, activate_codes_num):
    encoder_hidden = encoder.initHidden()
    # print("target_variable:", target_variable)    target_variable: [[-1], array([ 0,  1, 48, 49]), array([ 0,  1, 48, 49]), [-1]]
    
    #文本处理模型
    # txt_input = torch.LongTensor(list(txt_input)).cuda()
    # txt_outputs, txt_self_attns = txt_encoder(txt_input)
    # txt_outputs = txt_outputs.squeeze()
    # txt_outputs = txt_outputs.mean(dim=0)
    # txt_outputs = txt_outputs.unsqueeze(0)
    # txt_outputs = txt_outputs.unsqueeze(0)
    # # print("txt_outputs shape:", txt_outputs.shape)
    # encoder_hidden = txt_outputs

    input_length = len(input_variable)
    encoder_outputs = Variable(torch.zeros(MAX_LENGTH, encoder.hidden_size))
    if use_cuda:
        encoder_outputs = encoder_outputs.cuda()

    loss = 0

    history_record = np.zeros(output_size)
    count = 0
    for ei in range(input_length - 1):
        if ei == 0:
            continue
        for ele in input_variable[ei]:
            history_record[ele] += 1
        count += 1

    history_record = history_record / count

    for ei in range(input_length - 1):
        if ei == 0:
            continue
        encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)
        encoder_outputs[ei - 1] = encoder_output[0][0]

        for ii in range(k):
            vectorized_target = np.zeros(output_size)
            for idx in target_variable[ii + 1]:
                vectorized_target[idx] = 1

            vectorized_input = np.zeros(output_size)
            for idx in input_variable[ei]:
                vectorized_input[idx] = 1

    decoder_input = input_variable[input_length - 2]

    decoder_hidden = encoder_hidden
    last_hidden = decoder_hidden
    # Without teacher forcing: use its own predictions as the next input
    num_str = 0
    topk = 20
    decoded_vectors = []
    prob_vectors = []
    cout = 0
    for di in range(k):
        if atten_decoder:
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs, history_record, last_hidden)
        else:
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
        # print("*"*30)
        # print("decoder_output:", decoder_output, decoder_output.shape)
        topv, topi = decoder_output.data.topk(topk) #去topk得分最高的值和其对应的下标，topv为得分最高的topk个值，topi为其相应下标,topi.shape=(1,20)
        # print("topv:", topv, "topi:", topi, topi.shape)
        ni = topi[0][0]

        vectorized_target = np.zeros(output_size)
        for idx in target_variable[di + 1]:
            vectorized_target[idx] = 1  #根据真实值得到0,1向量，[0,0,0,1,0,0,1,1,0....]向量维度为所有item个数，相当于one-hot

        # target_topi = vectorized_target.argsort()[::-1][:topk]
        # activation_bound

        count = 0
        start_idx = -1
        end_idx = output_size
        if activate_codes_num > 0:
            pick_num = activate_codes_num
        else:
            pick_num = np.sum(vectorized_target)
            # print(pick_num)

        tmp = []
        for ele in range(len(topi[0])):
            if count >= pick_num:
                break
            tmp.append(topi[0][ele])
            count += 1

        decoded_vectors.append(tmp) #每个tmp的维度是实际限制的一个set里含有的item的个数
        decoder_input = tmp
        tmp = []
        for i in range(topk):
            tmp.append(topi[0][i])
        prob_vectors.append(tmp)    #每个tmp是topk维

    return decoded_vectors, prob_vectors


import torch.utils.bottleneck as bn


def top_n_indexes(arr, n):
    idx = bn.argpartition(arr, arr.size - n, axis=None)[-n:]
    width = arr.shape[1]
    return [divmod(i, width) for i in idx]


def get_precision_recall_Fscore(groundtruth, pred):
    a = groundtruth
    b = pred
    correct = 0
    truth = 0
    positive = 0

    for idx in range(len(a)):
        if a[idx] == 1:
            truth += 1
            if b[idx] == 1:
                correct += 1
        if b[idx] == 1:
            positive += 1

    flag = 0
    if 0 == positive:
        precision = 0
        flag = 1
        # print('postivie is 0')
    else:
        precision = correct / positive
    if 0 == truth:
        recall = 0
        flag = 1
        # print('recall is 0')
    else:
        recall = correct / truth

    if flag == 0 and precision + recall > 0:
        F = 2 * precision * recall / (precision + recall)
    else:
        F = 0
    return precision, recall, F, correct


def get_F_score(prediction, test_Y):
    jaccard_similarity = []
    prec = []
    rec = []

    count = 0
    for idx in range(len(test_Y)):
        pred = prediction[idx]
        T = 0
        P = 0
        correct = 0
        for id in range(len(pred)):
            if test_Y[idx][id] == 1:
                T = T + 1
                if pred[id] == 1:
                    correct = correct + 1
            if pred[id] == 1:
                P = P + 1

        if P == 0 or T == 0:
            continue
        precision = correct / P
        recall = correct / T
        prec.append(precision)
        rec.append(recall)
        if correct == 0:
            jaccard_similarity.append(0)
        else:
            jaccard_similarity.append(2 * precision * recall / (precision + recall))
        count = count + 1

    print(
        'average precision: ' + str(np.mean(prec)))
    print('average recall : ' + str(
        np.mean(rec)))
    print('average F score: ' + str(
        np.mean(jaccard_similarity)))


def get_DCG(groundtruth, pred_rank_list, k):
    count = 0
    dcg = 0
    for pred in pred_rank_list:
        if count >= k:
            break
        if groundtruth[pred] == 1:
            dcg += (1) / math.log2(count + 1 + 1)
        count += 1

    return dcg


def get_NDCG(groundtruth, pred_rank_list, k):
    count = 0
    dcg = 0
    for pred in pred_rank_list:
        if count >= k:
            break
        if groundtruth[pred] == 1:
            dcg += (1) / math.log2(count + 1 + 1)
        count += 1
    idcg = 0
    num_real_item = np.sum(groundtruth)
    num_item = int(min(num_real_item, k))
    for i in range(num_item):
        idcg += (1) / math.log2(i + 1 + 1)
    ndcg = dcg / idcg
    return ndcg


def get_HT(groundtruth, pred_rank_list, k):
    count = 0
    for pred in pred_rank_list:
        if count >= k:
            break
        if groundtruth[pred] == 1:
            return 1
        count += 1

    return 0


def evaluate(data_chunk, txt_inputs, txt_encoder, encoder, decoder, output_size, test_key_set, next_k_step, activate_codes_num):
    prec = []
    rec = []
    F = []
    prec1 = []
    rec1 = []
    F1 = []
    prec2 = []
    rec2 = []
    F2 = []
    prec3 = []
    rec3 = []
    F3 = []
    length = np.zeros(3)

    NDCG = []
    n_hit = 0
    count = 0

    for iter in range(len(test_key_set)):
        # training_pair = training_pairs[iter - 1]
        # input_variable = training_pair[0]
        # target_variable = training_pair[1]
        input_variable = data_chunk[past_chunk][test_key_set[iter]]
        target_variable = data_chunk[future_chunk][test_key_set[iter]]

        tmp_no = test_key_set[iter][4:]
        txt_input = txt_inputs[tmp_no]


        if len(target_variable) < 2 + next_k_step:
            continue
        count += 1
        output_vectors, prob_vectors = decoding_next_k_step(txt_input, txt_encoder, encoder, decoder, input_variable, target_variable,
                                                            output_size, next_k_step, activate_codes_num)

        hit = 0
        for idx in range(len(output_vectors)):
            # for idx in [2]:
            vectorized_target = np.zeros(output_size)
            for ii in target_variable[1 + idx]:
                vectorized_target[ii] = 1

            vectorized_output = np.zeros(output_size)
            for ii in output_vectors[idx]:
                vectorized_output[ii] = 1

            precision, recall, Fscore, correct = get_precision_recall_Fscore(vectorized_target, vectorized_output)
            prec.append(precision)
            rec.append(recall)
            F.append(Fscore)
            if idx == 0:
                prec1.append(precision)
                rec1.append(recall)
                F1.append(Fscore)
            elif idx == 1:
                prec2.append(precision)
                rec2.append(recall)
                F2.append(Fscore)
            elif idx == 2:
                prec3.append(precision)
                rec3.append(recall)
                F3.append(Fscore)
            length[idx] += np.sum(target_variable[1 + idx])
            target_topi = prob_vectors[idx]
            hit += get_HT(vectorized_target, target_topi, activate_codes_num)
            ndcg = get_NDCG(vectorized_target, target_topi, activate_codes_num)
            NDCG.append(ndcg)
        if hit == next_k_step:
            n_hit += 1

    print('average precision of subsequent sets' + ': ' + str(np.mean(prec)) + ' with std: ' + str(np.std(prec)))
    print('average recall' + ': ' + str(np.mean(rec)) + ' with std: ' + str(np.std(rec)))
    print('average F score of subsequent sets' + ': ' + str(np.mean(F)) + ' with std: ' + str(np.std(F)))
    # print('average precision of 1st' + ': ' + str(np.mean(prec1)) + ' with std: ' + str(np.std(prec1)))
    # print('average recall of 1st' + ': ' + str(np.mean(rec1)) + ' with std: ' + str(np.std(rec1)))
    # print('average F score of 1st' + ': ' + str(np.mean(F1)) + ' with std: ' + str(np.std(F1)))
    # print('average precision of 2nd' + ': ' + str(np.mean(prec2)) + ' with std: ' + str(np.std(prec2)))
    # print('average recall of 2nd' + ': ' + str(np.mean(rec2)) + ' with std: ' + str(np.std(rec2)))
    # print('average F score of 2nd' + ': ' + str(np.mean(F2)) + ' with std: ' + str(np.std(F2)))
    # print('average precision of 3rd' + ': ' + str(np.mean(prec3)) + ' with std: ' + str(np.std(prec3)))
    # print('average recall of 3rd' + ': ' + str(np.mean(rec3)) + ' with std: ' + str(np.std(rec3)))
    # print('average F score of 3rd' + ': ' + str(np.mean(F3)) + ' with std: ' + str(np.std(F3)))
    print('average NDCG: ' + str(np.mean(NDCG)))
    print('average hit rate: ' + str(n_hit / len(test_key_set)))

    return np.mean(rec), np.mean(NDCG), n_hit / len(test_key_set)

def partition_the_data(data_chunk, key_set, next_k_step):
    filtered_key_set = []
    for key in key_set:
        if len(data_chunk[past_chunk][key]) <= 3:
            continue
        if len(data_chunk[future_chunk][key]) < 2 + next_k_step:
            continue
        filtered_key_set.append(key)

    training_key_set = filtered_key_set[0:int(4 / 5 * len(filtered_key_set))]
    print('Number of training instances: ' + str(len(training_key_set)))
    test_key_set = filtered_key_set[int(4 / 5 * len(filtered_key_set)):]
    return training_key_set, test_key_set

def partition_the_data_validate(data_chunk, key_set, next_k_step):      #划分训练集、验证集、测试集
    filtered_key_set = []
    for key in key_set:
        if len(data_chunk[past_chunk][key]) <= 3:
            continue
        if len(data_chunk[future_chunk][key]) <  2+next_k_step: #因为开头和结尾各有一个-1，因此要加2
            continue
        filtered_key_set.append(key)
    training_key_set = filtered_key_set[0:int(4 / 5 * len(filtered_key_set)*0.9)]
    validation_key_set = filtered_key_set[int(4 / 5 * len(filtered_key_set)*0.9):int(4 / 5 * len(filtered_key_set))]
    print('Number of training instances: ' + str(len(training_key_set)))
    test_key_set = filtered_key_set[int(4 / 5 * len(filtered_key_set)):]
    return training_key_set, validation_key_set, test_key_set

def get_codes_frequency_no_vector(X, num_dim, key_set):
    result_vector = np.zeros(num_dim)

    for pid in key_set:
        for idx in X[pid]:
            result_vector[idx] += 1
    # print(len(result_vector))
    return result_vector

# The first two parameters are the past records and future records, respectively.
# The main function consists of two model which is decisded by the argv[5]. If training is 1, it is training mode. If
# training is 0, it is test mode. model_version is the name of the model. next_k_step is the number of steps we predict.
# model_epoch is the model generated by the model_epoch-th epoch.
def main():
    
    # files = [argv[1], argv[2]]
    # files = ['Dunnhumby_history_order_10_steps_50kuser.csv', 'Dunnhumby_future_order_10_steps_50kuser.csv']
    # files = ['Tmall_history_NB.csv', 'Tmall_future_NB.csv']

    #倒数三个参数依次是模型名字、所预测的后续集合个数、训练模式，训练模式为1表明是训练模式，训练模式为0表明是测试模式 
    # argv = ['Sets2Sets.py', './data/TaFang_history.csv', './data/TaFang_future.csv', 'TaFang', 1, 1]
    argv = ['Sets2Sets.py', './data/sample_train_cp_data.csv', './data/sample_test_cp_data.csv', 'without_diag_CP_diag_rec', 1, 0]

    files = ['TaFang_history.csv', 'TaFang_future.csv']
    model_version = 'Tafeng_0.001'

    files = [argv[1],argv[2]]

    model_version = argv[3]

    next_k_step = int(argv[4])  #所预测后续集合个数
    training = int(argv[5])     #训练模式
    path = './'
    directory = './models/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    #data_chunk:顾客和set对应，多个set构成一个list，每个set中的item保存成array, data_chunk是一个list， 由两个字典组成，分别是历史set和未来set
    #input_size:输入大小，向量维度加2（前后各有一个-1）
    #code_freq_at_first_claim:one-hot向量所有分量出现次数，初试为0
    data_chunk, input_size, code_freq_at_first_claim = read_claim2vector_embedding_file_no_vector(path, files)
    #past_chunk = 0, future_chunk = 1
    #codes_freq one-hot向量每个分量所代表的元素在历史set中出现的次数
    codes_freq = get_codes_frequency_no_vector(data_chunk[past_chunk], input_size, data_chunk[future_chunk].keys())
    # key_set只有住院号
    training_key_set, validation_key_set, test_key_set = partition_the_data_validate(data_chunk, list(data_chunk[future_chunk]), next_k_step)
    # print("*"*40)
    # print("training_key_set:", training_key_set)
    weights = np.zeros(input_size)
    max_freq = max(codes_freq)
    for idx in range(len(codes_freq)):
        if codes_freq[idx] > 0:
            weights[idx] = max_freq / codes_freq[idx]
        else:
            weights[idx] = 0
    #input_size:向量维度+2， hidden_size:32, num_layers = 1
    encoder1 = EncoderRNN_new(input_size, hidden_size, num_layers)
    attn_decoder1 = AttnDecoderRNN_new(hidden_size, input_size, num_layers, dropout_p=0.1)
    txt_inputs, txt_encoder = return_model_data()
    # summary(encoder1)
    # summary(attn_decoder1)
    # total_params1 = sum(p.numel() for p in encoder1.parameters())
    # total_params2 = sum(p.numel() for p in attn_decoder1.parameters())
    # print("num of parameters:", total_params1 + total_params2)

    if use_cuda:
        encoder1 = encoder1
        attn_decoder1 = attn_decoder1

    if training == 1:
        if atten_decoder:
            trainIters(txt_inputs, txt_encoder, data_chunk, input_size, encoder1, attn_decoder1, model_version, training_key_set, weights,
                       next_k_step, num_iter, print_every=print_val)

    else:
        for i in [10, 20, 30, 40]:
            valid_recall = []
            valid_ndcg = []
            valid_hr = []
            recall_list = []
            ndcg_list = []
            hr_list = []
            print('k = ' + str(i))
            for model_epoch in range(num_iter):
                print("*****************************************")
                print('Epoch: ', model_epoch)
                encoder_pathes = './models/encoder' + str(model_version) + '_model_epoch' + str(model_epoch)
                decoder_pathes = './models/decoder' + str(model_version) + '_model_epoch' + str(model_epoch)
                txt_encoder_pathes = './models/txt_encoder' + str(model_version) + '_model_epoch' + str(model_epoch)

                encoder_instance = torch.load(encoder_pathes)
                decoder_instance = torch.load(decoder_pathes)
                txt_encoder_instance = torch.load(txt_encoder_pathes)

                recall, ndcg, hr = evaluate(data_chunk, txt_inputs, txt_encoder_instance, encoder_instance, decoder_instance, input_size, validation_key_set, next_k_step, i)
                valid_recall.append(recall)
                valid_ndcg.append(ndcg)
                valid_hr.append(hr)
                recall, ndcg, hr = evaluate(data_chunk, txt_inputs, txt_encoder_instance, encoder_instance, decoder_instance, input_size, test_key_set, next_k_step, i)
                recall_list.append(recall)
                ndcg_list.append(ndcg)
                hr_list.append(hr)
            valid_recall = np.asarray(valid_recall)
            valid_ndcg = np.asarray(valid_ndcg)
            valid_hr = np.asarray(valid_hr)
            idx1 = valid_recall.argsort()[::-1][0]
            idx2 = valid_ndcg.argsort()[::-1][0]
            idx3 = valid_hr.argsort()[::-1][0]
            print('max valid recall results:')
            print('Epoch: ', idx1)
            print('recall: ', recall_list[idx1])
            print('ndcg: ', ndcg_list[idx1])
            print('phr: ', hr_list[idx1])

            print('max valid ndcg results:')
            print('Epoch: ', idx2)
            print('recall: ', recall_list[idx2])
            print('ndcg: ', ndcg_list[idx2])
            print('phr: ', hr_list[idx2])

            print('max valid phr results:')
            print('Epoch: ', idx3)
            print('recall: ', recall_list[idx3])
            print('ndcg: ', ndcg_list[idx3])
            print('phr: ', hr_list[idx3])

if __name__ == '__main__':
    main()

