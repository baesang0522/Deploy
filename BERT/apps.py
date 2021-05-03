from django.apps import AppConfig
from BERT.bert_testing import DataLoading

class BertConfig(AppConfig):
    model, device = DataLoading().data_load()
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'BERT'


