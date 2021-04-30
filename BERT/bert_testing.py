import os
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler
from transformers import BertForSequenceClassification
# from DB.models import HS6Label as hs6

batch_size = 8
output_dir = '.\\data'


def labeldata_2_dict(path):

    labels_dict = pd.read_csv(path, dtype=object, index_col=None)
    labels_dict = {labels_dict['labels'][idx]: int(labels_dict['numbers'][idx]) for idx in range(len(labels_dict))}
    id_to_label_dict = {value: key for key, value in labels_dict.items()}

    return labels_dict, id_to_label_dict


def prepare_testset(text):
    tokenizer = BertTokenizer.from_pretrained(output_dir)
    text = text.lower()
    test1 = {'eng_des': text}
    test_text = test1['eng_des']

    test_inputs = []
    test_am = []

    for sent in test_text:
        encoded_dict = tokenizer.encode_plus(
                                             sent,
                                             add_special_tokens=True,
                                             max_length=220,
                                             pad_to_max_length=True,
                                             return_attention_mask=True,
                                             return_tensors='pt',
                                             truncation=True
                                             )

        test_inputs.append(encoded_dict['input_ids'])
        test_am.append(encoded_dict['attention_mask'])

    test_inputs = torch.cat(test_inputs, dim=0)
    test_am = torch.cat(test_am, dim=0)

    test_dataset = TensorDataset(test_inputs, test_am)
    test_dataloader = DataLoader(test_dataset,
                                 sampler=RandomSampler(test_dataset),
                                 batch_size=batch_size)

    return test_dataset, test_dataloader


class DataLoading:

    def __init__(self):
        cwd = os.getcwd()
        where_label_csv = '\\data\\'
        label_csv_name = 'label_csv.csv'
        self.label_csv = cwd + where_label_csv + label_csv_name

    def data_load(self):
        labels_dict, id_to_label_dict = labeldata_2_dict(self.label_csv)
        trained_model = BertForSequenceClassification.from_pretrained(output_dir)

        if torch.cuda.is_available():
            DEVICE = torch.device('cuda')
            print('There are %d GPU(s) available.' % torch.cuda.device_count())
            print('We will use the GPU:', torch.cuda.get_device_name(0))
            print('batch_size:', batch_size)
        else:
            print('No GPU available, using the CPU instead.')
            print('batch_size:', batch_size)
            DEVICE = torch.device('cpu')

        trained_model.to(DEVICE)

        return labels_dict, id_to_label_dict, trained_model, DEVICE


def bert_predict(text, id_to_label_dict, trained_model, DEVICE):

    id_label_dict, model, device = id_to_label_dict, trained_model, DEVICE
    test_dataset, test_dataloader = prepare_testset(text)

    for batch in test_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)

        # tell pytorch not to train during val
        with torch.no_grad():
            logits = model(b_input_ids,
                           token_type_ids=None,
                           attention_mask=b_input_mask,
                           )[0]

        # Move logits and labels to CPU
        test_logits = logits.detach().cpu().numpy()

    top3_prediction = np.argpartition(-test_logits, 3)[0][:3].tolist()  # top3

    # if many, change to list type
    top1 = [id_label_dict[top3_prediction[0]]]
    top2 = [id_label_dict[top3_prediction[1]]]
    top3 = [id_label_dict[top3_prediction[2]]]

    result = {'top1': top1, 'top2': top2, 'top3': top3}

    return result


