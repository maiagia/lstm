from pydantic import BaseModel
from datetime import date

class ModeloRequestHistoricoPreco(BaseModel):
    """
    Modelo de entrada para solicitar o histórico de preços.
    """
    acao: str
    data_inicio: date
    data_fim: date