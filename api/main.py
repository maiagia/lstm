from fastapi import FastAPI
from rotas import predict, historico_preco

vApp = FastAPI(title="Redes Neurais - Long Short Term Memory (LSTM)")

vApp.include_router(router=predict.vRota, prefix='/api')
vApp.include_router(router=historico_preco.vRota, prefix='/api')