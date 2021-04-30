from django.apps import AppConfig
from BERT.bert_testing import DataLoading

class BertConfig(AppConfig):
    _, id_to_label, model, device = DataLoading().data_load()
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'BERT'


