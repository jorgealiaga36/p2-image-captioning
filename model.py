import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        self.word_embeds = nn.Embedding(vocab_size, embed_size)
        
        self.lstm = nn.LSTM(embed_size, 
                            hidden_size, 
                            num_layers, 
                            batch_first=True)
        
        self.linear_layer = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        embeddings = self.word_embeds(captions[:,:-1])
        inputs = torch.cat((features.unsqueeze(1), embeddings), 1)
        lstm_outputs = self.lstm(inputs)[0]
        outputs = self.linear_layer(lstm_outputs)
        
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        tokens = []
        
        for index in range(max_len):                                   
            lstm_outputs, states = self.lstm(inputs, states)      
            outputs = self.linear_layer(lstm_outputs.squeeze(1))
            predicted = outputs.max(1)[1]

            tokens.append(predicted.tolist()[0])
          
            inputs = self.word_embeds(predicted)
            inputs = inputs.unsqueeze(1)                     

        return tokens