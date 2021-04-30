from rest_framework.views import APIView
from rest_framework.response import Response
from .apps import BertConfig
from .bert_testing import bert_predict

# Create your views here.


class BertTestingApiView(APIView):
    def get(self, request, text):
        response = bert_predict(text=text,
                                id_to_label_dict=BertConfig.id_to_label,
                                trained_model=BertConfig.model,
                                DEVICE=BertConfig.device)

        return Response(response)