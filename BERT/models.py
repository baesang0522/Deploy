from django.db import models


class HS6Label(models.Model):
    labels = models.CharField(max_length=6, primary_key=True)
    ID = models.CharField(max_length=20000)

    class Meta:
        managed = True
        db_table = 'HS6_label'
