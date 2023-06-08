import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size):
        super(Encoder, self).__init__()
        self.embedding = nn.Sequential(
            nn.Embedding(vocab_size, embed_dim),
            nn.Dropout(0.5)
        )
        self.rnn = nn.GRU(embed_dim, hidden_size)

    def forward(self, inputs):
        outputs = self.embedding(inputs)
        outputs, hidden = self.rnn(outputs)
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, embed_dim, hidden_size):
        super(Attention, self).__init__()
        self.linear = nn.Linear(hidden_size + embed_dim, hidden_size)
        self.v = nn.Linear(hidden_size, 1)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, dec_input, enc_output):
        # enc_output 编码器outputs (seq len,batch,dim)
        # dec_input = decoder当前时间步的embed(batch,dim)
        seq_len = enc_output.shape[0]
        s = dec_input.repeat(seq_len, 1, 1)  # (seq len,batch,dim)
        x = torch.tanh(self.linear(torch.cat([enc_output, s], dim=2)))
        attention = self.v(x)
        return self.softmax(attention)  # [batch, seq_len]


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Sequential(
            nn.Embedding(vocab_size, embed_dim),
            nn.Dropout(0.5)
        )
        self.attention = Attention(100, 128)
        self.gru = nn.GRU(embed_dim + hidden_size, hidden_size)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim + 2 * hidden_size, vocab_size),
            nn.Dropout(0.5)

        )

    def forward(self, inputs, pre_hidden, context):
        # context : encoder outputs(seq len,batch,dim)  编码器的outputs，所有时间步的隐状态
        # pre_hidden:decoder pre_hidden (batch,dim)   解码器的上一个隐状态
        # embedded :precess text (batch,dim)  解码器embed输入
        embedded = self.embedding(inputs.unsqueeze(0))  # (seq len,batch)
        a = self.attention(embedded, context)  # 获取embed在context的权重
        c = torch.bmm(a, context)

        outputs, hidden = self.gru(torch.cat([embedded, c], dim=2), pre_hidden)
        outputs = self.fc(torch.cat([outputs.squeeze(0), embedded.squeeze(0), c.squeeze(0)], dim=-1))
        return outputs


import random


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, origin, target, isval=False, teacher_forcing_ratio=0.5):
        # origin : (seq len,batch)
        # target : (seq len,batch)
        batch_size = origin.shape[1]
        target_len = target.shape[0]
        target_vocab_size = self.decoder.vocab_size
        outputs = torch.zeros(target_len, batch_size, target_vocab_size)
        _, hidden = self.encoder(origin)
        context = hidden

        if not isval:
            # training ,inputs is true word or predict word
            top1 = 0
            for i in range(target_len):
                teacher_force = random.random() < teacher_forcing_ratio

                if i == 0:
                    teacher_force = True

                inputs = target[i] if teacher_force else top1

                output = self.decoder(inputs, hidden, context)
                # first input : encoder last outputs + target first seq
                outputs[i] = output
                top1 = output.argmax(1)
        else:
            # valing ,inputs is predict word
            inputs = target[0, :]
            for i in range(target_len):
                output = self.decoder(inputs, hidden, context)
                outputs[i] = output
                inputs = output.argmax(1)
        return outputs
