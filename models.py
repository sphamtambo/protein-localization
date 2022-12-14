from transformers import BertModel
import torch
import torch.nn.functional as F
import torch.nn as nn
from config import *

class ProteinClassifier(nn.Module):
    def __init__(self):
        super(ProteinClassifier, self).__init__()

        ### instantiate the bert model object
        self.bert = BertModel.from_pretrained(PRETAINED_MODEL)

        ### define layers
        self.classifier = nn.Sequential(
                nn.Dropout(DROPOUT_RATE),
                nn.Linear(self.bert.config.hidden_size, NUM_CLASSES),
                nn.Tanh()
                )

    def forward(self, input_ids, attention_mask):
        ### forward pass through pre-trained Bert
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        ### embedding of CLS token (from seq output) further processed by
        ### linear layer and a Tanh activation function
        return self.classifier(output.pooler_output)


class BertClassifier(nn.Module):

    def __init__(self, config):
        super(BertClassifier, self).__init__()
        ### instantiate the bert model object
        self.bert = BertModel.from_pretrained(PRETAINED_MODEL)
        ### dropout to avoid overfitting
        self.dropout = nn.Dropout(DROPOUT_RATE)
        ### single linear layer
        self.classifier = nn.Linear(HIDDEN_SIZE, NUM_CLASSES)
        ### weight initialization
        torch.nn.init.xavier_normal_(self.classifier.weight)
        ### activation fn used by BERT 
        self.tanh = nn.Tanh()

    def forward(self, input_ids, attention_mask):
        ### forward pass through pre-trained BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        ### last layer output (Total 12 layers)
        ### embedding of CLS token (from seq output) further processed by
        ### linear layer and a Tanh activation function
        pooled_output = outputs[-1]
        dropout_output = self.dropout(pooled_output)
        linear_output = self.classifier(pooled_output)
        proba = self.tanh(pooled_output)
        return proba


