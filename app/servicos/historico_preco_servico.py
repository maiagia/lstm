import yfinance as yf

def obter_historico_preco(pAcao: str, pDataInicio: str, pDataFim: str):
    """
    Baixa o histórico de preços da ação informada entre as datas de início e fim.
    """
    vBaseHistorico = yf.download(pAcao, start=pDataInicio, end=pDataFim).reset_index().droplevel(axis=1, level=1)
    vBaseHistorico['Acao'] = pAcao
    vBaseHistorico_Dict = vBaseHistorico.to_dict(orient='records')
    return vBaseHistorico_Dict

# print(obter_historico_preco("AAPL", "2022-01-01", "2022-01-05"))