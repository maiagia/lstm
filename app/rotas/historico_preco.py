from fastapi import APIRouter
from typing import List, Dict
from modelos.historico_preco_modelo import ModeloRequestHistoricoPreco
from servicos.historico_preco_servico import obter_historico_preco

vRota = APIRouter()

@vRota.post("/historico_preco", response_model=List[Dict])
def historicoPrecoEndpoint(vModeloRequestHistoricoPreco: ModeloRequestHistoricoPreco):
    """
    Endpoint para obter o histórico de preços de uma ação.
    """
    vResultado = obter_historico_preco(pAcao=vModeloRequestHistoricoPreco.acao,
                                       pDataInicio=str(vModeloRequestHistoricoPreco.data_inicio),
                                       pDataFim=str(vModeloRequestHistoricoPreco.data_fim))
    return vResultado