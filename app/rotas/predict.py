from fastapi import APIRouter
from typing import Dict
from modelos.request_preco_modelo import ModeloRequestPreco
from servicos.predict_servico import prever_precos

vRota = APIRouter()

@vRota.post("/predict")
def predictEndpoint(pListaPrecosAnteriores: ModeloRequestPreco):
    """
    Endpoint para prever o preço futuro com base nos preços anteriores.
    """
    vResultado = prever_precos(pListaPrecosAnteriores=pListaPrecosAnteriores.precos_anteriores)
    return {vResultado}