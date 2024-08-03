import torch
import numpy as np
from torch import optim, nn
from meta_learning.loss_function import *


class lstm_encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):

        super(lstm_encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers)

    def forward(self, x_input):
        lstm_out, self.hidden = self.lstm(x_input.view(x_input.shape[0], x_input.shape[1], self.input_size))

        return lstm_out, self.hidden

    def init_hidden(self, batch_size):

        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))


class lstm_decoder(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1):

        super(lstm_decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers)

        self.linear = nn.Linear(hidden_size, input_size)

    def forward(self, x_input, encoder_hidden_states):

        lstm_out, self.hidden = self.lstm(x_input.unsqueeze(0), encoder_hidden_states)
        output = self.linear(lstm_out.squeeze(0))

        return output, self.hidden


class lstm_seq2seq(nn.Module):
    def __init__(self, input_size, hidden_size):

        super(lstm_seq2seq, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.encoder = lstm_encoder(input_size=input_size, hidden_size=hidden_size)
        self.decoder = lstm_decoder(input_size=input_size, hidden_size=hidden_size)


    def train_model_meta_adapt(self, input_tensor, target_tensor, target_len):

        criterion = nn.MSELoss()

        batch_size = input_tensor.shape[1]
        n_batches = int(input_tensor.shape[1] / batch_size)

        batch_loss = 0.

        for b in range(n_batches):
            input_batch = input_tensor[:, b: b + batch_size, :]
            target_batch = target_tensor[:, b: b + batch_size, :]

            outputs = torch.zeros(target_len, batch_size, input_batch.shape[2])

            encoder_hidden = self.encoder.init_hidden(batch_size)

            encoder_output, encoder_hidden = self.encoder(input_batch)

            decoder_input = input_batch[-1, :, :]  # shape: (batch_size, input_size)
            decoder_hidden = encoder_hidden

            for t in range(target_len):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                outputs[t] = decoder_output
                decoder_input = decoder_output

            loss = criterion(outputs.to("cuda:0"), target_batch)
            loss.requires_grad_(True)
            batch_loss += loss

        batch_loss /= n_batches
        return batch_loss

    def train_model_learning_path_construction(self, input_tensor, target_tensor, n_epochs, target_len, path, learning_rate=0.01):

        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        batch_size = input_tensor.shape[1]
        n_batches =int(input_tensor.shape[1] / batch_size)

        learning_path = []
        for epoch in range(n_epochs):
            for b in range(n_batches):

                input_batch = input_tensor[:, b: b + batch_size, :]
                target_batch = target_tensor[:, b: b + batch_size, :]

                outputs = torch.zeros(target_len, batch_size, input_batch.shape[2])

                encoder_hidden = self.encoder.init_hidden(batch_size)

                optimizer.zero_grad()

                encoder_output, encoder_hidden = self.encoder(input_batch)


                decoder_input = input_batch[-1, :, :]  # shape: (batch_size, input_size)
                decoder_hidden = encoder_hidden

                for t in range(target_len):
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                    outputs[t] = decoder_output
                    decoder_input = decoder_output

                loss = criterion(outputs.to("cuda:0"), target_batch)
                loss.requires_grad_(True)

            loss.backward()
            optimizer.step()
            grad = []
            for name, parms in self.named_parameters():
                this_grad = np.array(parms.grad.cpu().flatten()).tolist()
                grad = grad + this_grad
            learning_path.append(grad)

        np_learning_path = np.array(learning_path)
        np.savetxt(path + "learning_path.csv", np_learning_path, delimiter=",")
        return None
