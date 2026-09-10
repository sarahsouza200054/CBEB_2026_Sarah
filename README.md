# Habitats por Grid Regular

Divide o tumor em blocos KxKxK, extrai features radiômicas, agrupa os blocos em habitats (K-means) e mede a estabilidade dos habitats ao variar os parâmetros.

## Arquivos

Coloque todos na mesma pasta:

- `Habitats.py`
- `Features.py`
- `Metricas.py`
- `RodarPipelines.ipynb`

## Instalação

```bash
pip install numpy pandas scipy scikit-learn SimpleITK matplotlib jupyter
```

## Como rodar

1. Coloque as imagens e as máscaras em uma pasta (por exemplo, `Casos/`).
2. Abra `RodarPipelines.ipynb` e edite a célula **EDITE AQUI**:
   - pastas de entrada e saída;
   - lista de casos (imagem e máscara de cada um);
   - lista de configurações. A primeira linha é a referência.
3. Execute todas as células.

## Saídas

- `Saidas/<configuração>/`: rótulos, centroides e parâmetros de cada execução.
- `Visualizacoes/<configuração>/`: um PDF por caso.
- `Saidas/metricas_estabilidade.csv`: tabela de estabilidade.
