import transformers
from transformers import BertModel, BertConfig
import torch.nn as nn

class Mymodel(nn.Module):
    def __init__(self, classes=4):
        super(Mymodel, self).__init__()
        self.config = BertConfig.from_pretrained('bert-base-cased')
        self.config.update({'output_hidden_states': True})
        self.bert = BertModel.from_pretrained("bert-base-cased", config=self.config)
        self.linear = nn.Linear(768, classes)

    def forward(self, input_ids, attention_mask, token_type_ids,):
        output = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        output = output['pooler_output']
        output = self.linear(output)
        return output

