import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score


def _validar_mesma_shape(array_a, array_b, nome_a: str, nome_b: str):
    '''
    Auxiliar interna: levanta ValueError se os dois arrays não tiverem a mesma
    shape. Centraliza a checagem repetida nas funções de métrica deste arquivo.
    '''
    if array_a.shape != array_b.shape:
        raise ValueError(
            f"{nome_a} e {nome_b} devem ter a mesma shape "
            f"({array_a.shape} != {array_b.shape})."
        )


def _concatenar_intersecoes(volumes_ref: list, volumes_teste: list):
    '''
    Auxiliar interna: percorre os casos, seleciona os voxels da interseção
    (rótulo != 0 nos dois volumes) e devolve os rótulos de referência e de teste
    concatenados de todos os casos. Casos com interseção vazia são ignorados.

    Saída:
        - (rotulos_ref, rotulos_teste): dois np.ndarray 1D de mesmo tamanho, ou
          (None, None) se nenhum caso contribuiu voxels.
    '''
    if len(volumes_ref) != len(volumes_teste):
        raise ValueError(
            f"volumes_ref e volumes_teste devem ter o mesmo número de casos "
            f"({len(volumes_ref)} != {len(volumes_teste)})."
        )

    rotulos_ref_concatenados = []
    rotulos_teste_concatenados = []

    for volume_ref, volume_teste in zip(volumes_ref, volumes_teste):
        volume_ref = np.asarray(volume_ref)
        volume_teste = np.asarray(volume_teste)
        _validar_mesma_shape(volume_ref, volume_teste, 'volume_ref', 'volume_teste')

        mascara_intersecao = (volume_ref != 0) & (volume_teste != 0)
        if not np.any(mascara_intersecao):
            continue

        rotulos_ref_concatenados.append(volume_ref[mascara_intersecao])
        rotulos_teste_concatenados.append(volume_teste[mascara_intersecao])

    if not rotulos_ref_concatenados:
        return None, None

    return np.concatenate(rotulos_ref_concatenados), np.concatenate(rotulos_teste_concatenados)


def _pareamento_reverso_e_rotulos_ref(volume_ref, pareamento: dict):
    '''
    Auxiliar interna: inverte o dict de pareamento (rotulo_ref -> rotulo_teste),
    seguro porque a correspondência vinda de linear_sum_assignment é 1-para-1, e
    monta o universo ORDENADO de rótulos de referência — os presentes neste caso
    mais os mencionados no pareamento, para cobrir tanto "sem par" quanto
    "ausente neste volume".
    '''
    rotulos_ref_para_teste = {rotulo_ref: rotulo_teste for rotulo_teste, rotulo_ref in pareamento.items()}
    rotulos_ref_universo = sorted(
        set(np.unique(volume_ref[volume_ref != 0]).tolist()) | set(pareamento.values())
    )
    return rotulos_ref_para_teste, rotulos_ref_universo


def ari_habitats(volume_ref, volume_teste):
    '''
    Calcula o Adjusted Rand Index (ARI) entre dois volumes de rótulos de
    habitat, restrito à interseção dos voxels rotulados (rótulo != 0) nos
    dois volumes.

    Convenção dos volumes: 0 = voxel sem rótulo (fora da máscara, ou
    supervoxel descartado/não coberto); 1..k = rótulo do habitat. Como a
    cobertura de cada execução depende dos parâmetros usados (kernel, passo,
    raio, etc.), os dois volumes podem ter conjuntos de voxels rotulados
    diferentes — por isso a comparação é feita apenas sobre a interseção, e
    não sobre o volume inteiro.

    Entradas:
        - volume_ref: np.ndarray com os rótulos de referência.
        - volume_teste: np.ndarray com os rótulos da execução a comparar
          (mesma shape de volume_ref).

    Saída:
        - float com o ARI entre os dois rotulamentos na interseção, ou
          np.nan se a interseção for vazia (nenhum voxel rotulado em comum
          nos dois volumes).
    '''

    volume_ref = np.asarray(volume_ref)
    volume_teste = np.asarray(volume_teste)
    _validar_mesma_shape(volume_ref, volume_teste, 'volume_ref', 'volume_teste')

    mascara_intersecao = (volume_ref != 0) & (volume_teste != 0)

    if not np.any(mascara_intersecao):
        return np.nan

    return adjusted_rand_score(volume_ref[mascara_intersecao], volume_teste[mascara_intersecao])


def ari_habitats_conjunto(volumes_ref: list, volumes_teste: list):
    '''
    Versão conjunta de ari_habitats(): concatena os voxels da interseção
    (rótulo != 0 nos dois volumes) de todos os casos e calcula um único ARI
    para a execução inteira, em vez de um ARI por caso.

    Diferença em relação a ari_habitats(): dentro de uma execução, o K-means
    é ajustado uma única vez sobre a concatenação das features dos casos, de
    modo que um mesmo rótulo significa o mesmo habitat em todos os casos
    daquela execução. Por isso, a estabilidade entre duas execuções deve ser
    avaliada globalmente (rótulos de todos os casos concatenados), e não
    caso a caso — um ARI por caso ignoraria essa correspondência
    compartilhada entre os casos.

    Casos cujo volume de referência ou de teste esteja inteiramente sem
    rótulos (interseção vazia) são ignorados na concatenação, e não causam
    erro.

    Entradas:
        - volumes_ref: list[np.ndarray] — volumes de rótulos de referência,
          um por caso.
        - volumes_teste: list[np.ndarray] — volumes de rótulos da execução a
          comparar, um por caso, na mesma ordem de volumes_ref.

    Saída:
        - float com o ARI calculado sobre a concatenação dos voxels da
          interseção de todos os casos, ou np.nan se nenhum caso contribuiu
          voxels (todas as interseções vazias).
    '''

    rotulos_ref_concatenados, rotulos_teste_concatenados = _concatenar_intersecoes(volumes_ref, volumes_teste)

    if rotulos_ref_concatenados is None:
        return np.nan

    return adjusted_rand_score(rotulos_ref_concatenados, rotulos_teste_concatenados)


def parear_rotulos_global(volumes_ref: list, volumes_teste: list) -> dict:
    '''
    Encontra a correspondência ótima entre os rótulos da execução de teste e
    os rótulos da execução de referência, calculada UMA ÚNICA VEZ para os 4
    casos juntos (pareamento global por execução), e não caso a caso.

    Isso é necessário porque, dentro de uma execução, o K-means é ajustado
    uma única vez sobre a concatenação das features dos 4 casos — um mesmo
    rótulo (ex.: 2) significa o mesmo habitat nos quatro casos daquela
    execução. Entre execuções diferentes (parâmetros diferentes), os ajustes
    são independentes e os índices dos rótulos são arbitrários, então é
    preciso descobrir qual rótulo da referência corresponde a qual rótulo do
    teste. Um pareamento feito separadamente por caso poderia atribuir
    correspondências diferentes para casos da mesma execução, contradizendo
    o fato de os rótulos serem compartilhados — por isso os voxels dos 4
    casos são acumulados em uma única matriz de contingência antes de
    resolver o pareamento.

    O pareamento é resolvido com scipy.optimize.linear_sum_assignment sobre
    a matriz de contingência (rótulo de referência x rótulo de teste),
    maximizando o total de voxels coincidentes.

    Entradas:
        - volumes_ref: list[np.ndarray] — volumes de rótulos de referência,
          um por caso.
        - volumes_teste: list[np.ndarray] — volumes de rótulos da execução a
          comparar, um por caso, na mesma ordem de volumes_ref.

    Saída:
        - dict {rotulo_teste: rotulo_ref_correspondente}. Se o número de
          rótulos presentes na referência e no teste for diferente, os
          rótulos sem par ficam de fora do dict. Se nenhuma interseção
          existir em nenhum caso, retorna um dict vazio.
    '''

    rotulos_ref_concatenados, rotulos_teste_concatenados = _concatenar_intersecoes(volumes_ref, volumes_teste)

    if rotulos_ref_concatenados is None:
        return {}

    # Matriz de contingência única, acumulada sobre os voxels da interseção dos 4 casos.
    rotulos_ref_unicos = np.unique(rotulos_ref_concatenados)
    rotulos_teste_unicos = np.unique(rotulos_teste_concatenados)

    indices_ref = np.searchsorted(rotulos_ref_unicos, rotulos_ref_concatenados)
    indices_teste = np.searchsorted(rotulos_teste_unicos, rotulos_teste_concatenados)

    matriz_contingencia = np.zeros((len(rotulos_ref_unicos), len(rotulos_teste_unicos)), dtype=np.int64)
    np.add.at(matriz_contingencia, (indices_ref, indices_teste), 1)

    # linear_sum_assignment minimiza custo; negativa a contingência para maximizar coincidências.
    # Suporta matriz não quadrada, casando apenas min(n_ref, n_teste) pares.
    linhas, colunas = linear_sum_assignment(-matriz_contingencia)

    return {
        int(rotulos_teste_unicos[coluna]): int(rotulos_ref_unicos[linha])
        for linha, coluna in zip(linhas, colunas)
    }


def concordancia_voxels(volume_ref, volume_teste, pareamento: dict) -> float:
    '''
    Mede a fração de voxels da interseção (rótulo != 0 nos dois volumes) de
    UM caso em que o rótulo do teste, já traduzido pelo pareamento GLOBAL da
    execução (ver parear_rotulos_global()), coincide com o rótulo de
    referência.

    O pareamento é recebido pronto — esta função não recalcula
    correspondência de rótulos, apenas aplica o dict fornecido, para que a
    mesma tradução de rótulos seja usada de forma consistente nos 4 casos de
    uma execução.

    Entradas:
        - volume_ref: np.ndarray com os rótulos de referência de um caso.
        - volume_teste: np.ndarray com os rótulos de teste do mesmo caso
          (mesma shape de volume_ref).
        - pareamento: dict {rotulo_teste: rotulo_ref_correspondente}, obtido
          de parear_rotulos_global() para a execução inteira.

    Saída:
        - float entre 0 e 1 com a fração de voxels da interseção em que o
          rótulo pareado do teste coincide com o rótulo de referência.
          Rótulos de teste ausentes do pareamento contam como discordância.
          Retorna np.nan se a interseção for vazia.
    '''

    volume_ref = np.asarray(volume_ref)
    volume_teste = np.asarray(volume_teste)
    _validar_mesma_shape(volume_ref, volume_teste, 'volume_ref', 'volume_teste')

    mascara_intersecao = (volume_ref != 0) & (volume_teste != 0)
    if not np.any(mascara_intersecao):
        return np.nan

    rotulos_ref = volume_ref[mascara_intersecao]
    rotulos_teste = volume_teste[mascara_intersecao]

    # Traduz apenas os rótulos de teste distintos (poucos) e propaga para todos os voxels.
    # -1 como sentinela: nunca coincide com um rótulo de referência (sempre >= 1 na interseção).
    rotulos_teste_unicos, indices_inversos = np.unique(rotulos_teste, return_inverse=True)
    traducao = np.array([pareamento.get(int(rotulo), -1) for rotulo in rotulos_teste_unicos])
    rotulos_teste_traduzidos = traducao[indices_inversos]

    return float(np.mean(rotulos_teste_traduzidos == rotulos_ref))


def deslocamento_centroides_espacial(volume_ref, volume_teste, pareamento: dict, spacing=None) -> dict:
    '''
    Deslocamento do centroide ESPACIAL de cada habitat entre duas execuções,
    para UM caso.

    Centroide espacial: a posição média (z, y, x) dos voxels de um rótulo,
    dentro do volume desse caso — não deve ser confundido com o centroide no
    ESPAÇO DE CARACTERÍSTICAS (ver deslocamento_centroides_features()), que é
    a posição do cluster no espaço das features padronizadas e é único por
    execução, não por caso. Aqui, a média é calculada sobre o volume
    INTEIRO do caso (não apenas na interseção entre volume_ref e
    volume_teste), pois cada rótulo é bem definido dentro do seu próprio
    volume.

    Usa o dict de pareamento GLOBAL já calculado por parear_rotulos_global()
    para traduzir rótulos entre as duas execuções — não recalcula
    pareamento internamente.

    Entradas:
        - volume_ref: np.ndarray com os rótulos de referência de um caso
          (mesma shape de volume_teste).
        - volume_teste: np.ndarray com os rótulos de teste do mesmo caso.
        - pareamento: dict {rotulo_teste: rotulo_ref_correspondente}, obtido
          de parear_rotulos_global() para a execução inteira.
        - spacing: tupla (dz, dy, dx) com o espaçamento físico dos voxels.
          Se fornecido, a diferença de coordenadas é multiplicada por
          spacing e a distância retornada é em milímetros; se None, a
          distância é em voxels.

    Saída:
        - dict {rotulo_ref: distancia_euclidiana}, para todo rotulo_ref
          presente no volume_ref ou mencionado no pareamento. Rótulos sem
          par no pareamento, ou cujo rótulo (de referência ou de teste,
          conforme o caso) não tenha nenhum voxel no respectivo volume,
          recebem np.nan.
    '''

    volume_ref = np.asarray(volume_ref)
    volume_teste = np.asarray(volume_teste)
    _validar_mesma_shape(volume_ref, volume_teste, 'volume_ref', 'volume_teste')

    rotulos_ref_para_teste, rotulos_ref_universo = _pareamento_reverso_e_rotulos_ref(volume_ref, pareamento)

    resultado = {}
    for rotulo_ref in rotulos_ref_universo:
        rotulo_teste = rotulos_ref_para_teste.get(rotulo_ref)
        if rotulo_teste is None:
            resultado[rotulo_ref] = np.nan
            continue

        coords_ref = np.argwhere(volume_ref == rotulo_ref)
        coords_teste = np.argwhere(volume_teste == rotulo_teste)

        if coords_ref.size == 0 or coords_teste.size == 0:
            resultado[rotulo_ref] = np.nan
            continue

        diferenca = coords_teste.mean(axis=0) - coords_ref.mean(axis=0)
        if spacing is not None:
            diferenca = diferenca * np.asarray(spacing, dtype=float)

        resultado[rotulo_ref] = float(np.linalg.norm(diferenca))

    return resultado


def deslocamento_centroides_features(centros_ref, centros_teste, pareamento: dict) -> dict:
    '''
    Deslocamento do centroide no ESPAÇO DE CARACTERÍSTICAS de cada habitat
    entre duas execuções.

    Centroide de features: uma linha de cluster_centers_ do K-means, ajustado
    uma única vez sobre os 4 casos concatenados de uma execução — portanto é
    único POR EXECUÇÃO, e não por caso, ao contrário do centroide ESPACIAL
    (ver deslocamento_centroides_espacial()), que é calculado por caso a
    partir das coordenadas dos voxels.

    AVISO: essa distância só é interpretável se as duas execuções usarem o
    mesmo conjunto de features, na mesma ordem, e o mesmo procedimento de
    padronização. Como o StandardScaler é reajustado (fit) em cada execução
    — sobre dados potencialmente diferentes —, os dois espaços padronizados
    não são idênticos; a distância deve ser lida apenas como indicativo de
    deslocamento, não como uma medida absoluta comparável entre execuções
    com features ou pré-processamento diferentes.

    Os rótulos do pareamento seguem a convenção dos volumes de rótulo
    (1..k); como cluster_centers_ é indexado pelo Cluster_Label bruto
    (0..k-1), a linha usada para o rótulo `r` é `r - 1`.

    Entradas:
        - centros_ref: np.ndarray (k_ref, n_features) — cluster_centers_ da
          execução de referência.
        - centros_teste: np.ndarray (k_teste, n_features) — cluster_centers_
          da execução de teste.
        - pareamento: dict {rotulo_teste: rotulo_ref_correspondente}, obtido
          de parear_rotulos_global() para a execução inteira.

    Saída:
        - dict {rotulo_ref: distancia_euclidiana} entre o centroide de
          features pareado de teste e o de referência, para cada par do
          pareamento. Um rótulo cujo índice (rotulo - 1) esteja fora dos
          limites de centros_ref ou centros_teste recebe np.nan.
    '''

    centros_ref = np.asarray(centros_ref)
    centros_teste = np.asarray(centros_teste)

    if centros_ref.shape[1] != centros_teste.shape[1]:
        raise ValueError(
            f"centros_ref e centros_teste devem ter o mesmo número de features "
            f"({centros_ref.shape[1]} != {centros_teste.shape[1]})."
        )

    resultado = {}
    for rotulo_teste, rotulo_ref in pareamento.items():
        indice_ref = rotulo_ref - 1
        indice_teste = rotulo_teste - 1

        if not (0 <= indice_ref < centros_ref.shape[0]) or not (0 <= indice_teste < centros_teste.shape[0]):
            resultado[rotulo_ref] = np.nan
            continue

        diferenca = centros_teste[indice_teste] - centros_ref[indice_ref]
        resultado[rotulo_ref] = float(np.linalg.norm(diferenca))

    return resultado


def variacao_volume_habitats(volume_ref, volume_teste, pareamento: dict) -> pd.DataFrame:
    '''
    Compara o volume (número de voxels) de cada habitat entre duas execuções,
    para UM caso.

    A contagem de voxels de cada rótulo é feita sobre o volume INTEIRO do
    caso (não apenas na interseção entre volume_ref e volume_teste), pois
    cada rótulo é bem definido dentro do seu próprio volume. Usa o dict de
    pareamento GLOBAL já calculado por parear_rotulos_global() para
    traduzir rótulos entre as duas execuções — não recalcula pareamento
    internamente.

    Como o K-means é ajustado sobre os 4 casos concatenados de uma execução,
    um caso individual pode não conter todos os k habitats: tanto um
    habitat de referência sem par no pareamento quanto um habitat presente
    no pareamento mas ausente deste volume aparecem com voxels_teste = 0
    (ou voxels_ref = 0), sem levantar exceção.

    Entradas:
        - volume_ref: np.ndarray com os rótulos de referência de um caso
          (mesma shape de volume_teste).
        - volume_teste: np.ndarray com os rótulos de teste do mesmo caso.
        - pareamento: dict {rotulo_teste: rotulo_ref_correspondente}, obtido
          de parear_rotulos_global() para a execução inteira.

    Saída:
        - pd.DataFrame com uma linha por habitat de referência (todo
          rotulo_ref presente em volume_ref ou mencionado no pareamento) e
          as colunas:
            - rotulo_ref, rotulo_teste (np.nan se o habitat de referência
              não tiver par no pareamento);
            - voxels_ref, voxels_teste: contagem de voxels em cada volume;
            - variacao_absoluta: voxels_teste - voxels_ref;
            - variacao_percentual: variacao_absoluta em % de voxels_ref,
              ou np.nan se voxels_ref for 0 (evita divisão por zero);
            - proporcao_ref, proporcao_teste: voxels_ref / total de voxels
              rotulados em volume_ref, e o equivalente para volume_teste
              (np.nan se o respectivo total for 0).
    '''

    volume_ref = np.asarray(volume_ref)
    volume_teste = np.asarray(volume_teste)
    _validar_mesma_shape(volume_ref, volume_teste, 'volume_ref', 'volume_teste')

    colunas = ['rotulo_ref', 'rotulo_teste', 'voxels_ref', 'voxels_teste',
               'variacao_absoluta', 'variacao_percentual', 'proporcao_ref', 'proporcao_teste']

    rotulos_ref_para_teste, rotulos_ref_universo = _pareamento_reverso_e_rotulos_ref(volume_ref, pareamento)

    if not rotulos_ref_universo:
        return pd.DataFrame(columns=colunas)

    total_ref = int(np.count_nonzero(volume_ref))
    total_teste = int(np.count_nonzero(volume_teste))

    linhas = []
    for rotulo_ref in rotulos_ref_universo:
        rotulo_teste = rotulos_ref_para_teste.get(rotulo_ref)

        voxels_ref = int(np.count_nonzero(volume_ref == rotulo_ref))
        voxels_teste = int(np.count_nonzero(volume_teste == rotulo_teste)) if rotulo_teste is not None else 0

        variacao_absoluta = voxels_teste - voxels_ref
        variacao_percentual = np.nan if voxels_ref == 0 else (variacao_absoluta / voxels_ref) * 100.0

        proporcao_ref = np.nan if total_ref == 0 else voxels_ref / total_ref
        proporcao_teste = np.nan if total_teste == 0 else voxels_teste / total_teste

        linhas.append({
            'rotulo_ref': rotulo_ref,
            'rotulo_teste': rotulo_teste if rotulo_teste is not None else np.nan,
            'voxels_ref': voxels_ref,
            'voxels_teste': voxels_teste,
            'variacao_absoluta': variacao_absoluta,
            'variacao_percentual': variacao_percentual,
            'proporcao_ref': proporcao_ref,
            'proporcao_teste': proporcao_teste,
        })

    return pd.DataFrame(linhas, columns=colunas)


def voxels_perdidos(volume_rotulos, mascara) -> dict:
    '''
    Quantifica, para UM caso, quantos voxels do tumor original ficaram sem
    rótulo de habitat no pipeline. Métrica por caso, independente de
    qualquer pareamento de rótulos entre execuções (não usa
    parear_rotulos_global()).

    Um voxel do tumor pode ficar sem rótulo (volume_rotulos == 0) por três
    motivos: está fora da máscara fechada (a máscara ORIGINAL, antes do
    fechamento morfológico, é maior ou diferente da máscara usada na
    extração); pertence a um supervoxel descartado em extrair() por estar
    na borda do volume ou por não reunir kernel**3 voxels válidos; ou não
    foi coberto por nenhum bloco do grid regular (ocorre quando passo >
    kernel, deixando lacunas entre os blocos).

    O denominador usado no percentual é voxels_tumor: o número de voxels da
    MÁSCARA ORIGINAL (mascara > 0), lida diretamente de mascara_i.mha, antes
    do fechamento morfológico feito em fechar_esfera(). Não é o total de
    voxels rotulados nem o volume do grid regular.

    Entradas:
        - volume_rotulos: np.ndarray com os rótulos de habitat de um caso
          (0 = sem rótulo; 1..k = habitat), mesma shape de mascara.
        - mascara: np.ndarray da máscara do tumor original (antes do
          fechamento morfológico), lida de mascara_i.mha. Voxels com
          valor > 0 são considerados tumor.

    Saída:
        - dict com:
            - voxels_tumor: voxels da máscara original (mascara > 0).
            - voxels_rotulados: voxels do tumor (dentro da máscara
              original) com rótulo != 0.
            - voxels_perdidos: voxels_tumor - voxels_rotulados.
            - percentual_perdido: voxels_perdidos / voxels_tumor * 100, ou
              np.nan se voxels_tumor for 0.
            - voxels_rotulados_fora_mascara: voxels com rótulo != 0 fora da
              máscara original — efeito possível do fechamento morfológico
              ter incluído voxels que a máscara original não continha.
    '''

    volume_rotulos = np.asarray(volume_rotulos)
    mascara = np.asarray(mascara)
    _validar_mesma_shape(volume_rotulos, mascara, 'volume_rotulos', 'mascara')

    mascara_tumor = mascara > 0
    volume_rotulado = volume_rotulos != 0

    voxels_tumor = int(np.count_nonzero(mascara_tumor))
    voxels_rotulados = int(np.count_nonzero(volume_rotulado & mascara_tumor))
    voxels_perdidos_contagem = voxels_tumor - voxels_rotulados
    percentual_perdido = np.nan if voxels_tumor == 0 else (voxels_perdidos_contagem / voxels_tumor) * 100.0
    voxels_rotulados_fora_mascara = int(np.count_nonzero(volume_rotulado & ~mascara_tumor))

    return {
        'voxels_tumor': voxels_tumor,
        'voxels_rotulados': voxels_rotulados,
        'voxels_perdidos': voxels_perdidos_contagem,
        'percentual_perdido': percentual_perdido,
        'voxels_rotulados_fora_mascara': voxels_rotulados_fora_mascara,
    }


def _media_e_maximo_sem_nan(valores):
    '''
    Auxiliar interna: média e máximo de uma sequência de floats, ignorando
    np.nan, sem disparar os RuntimeWarning de np.nanmean/np.nanmax quando
    todos os valores (ou a sequência) são vazios/nan.
    '''
    validos = [v for v in valores if not np.isnan(v)]
    if not validos:
        return np.nan, np.nan
    return float(np.mean(validos)), float(np.max(validos))


def tabela_estabilidade_execucao(volumes_ref, volumes_teste, mascaras, centros_ref, centros_teste, metadados: dict, spacing=None) -> pd.DataFrame:
    '''
    Monta a tabela de estabilidade de UMA execução de teste contra a
    execução de referência, com uma linha por caso, reunindo todas as
    métricas já implementadas neste arquivo.

    O pareamento de rótulos (parear_rotulos_global) é GLOBAL POR EXECUÇÃO:
    calculado UMA ÚNICA VEZ sobre todos os casos e reutilizado em toda
    métrica que dependa dele (concordancia_voxels,
    deslocamento_centroides_espacial, deslocamento_centroides_features,
    variacao_volume_habitats) — nunca recalculado por caso, para não
    quebrar a correspondência de rótulos compartilhada entre os casos de
    uma mesma execução.

    Duas métricas são POR EXECUÇÃO, não por caso — porque dependem do
    ajuste conjunto do K-means sobre os 4 casos concatenados (ari_conjunto)
    ou dos cluster_centers_, que também são únicos por execução
    (deslocamento_centroide_features_medio/max) — e por isso aparecem
    repetidas em todas as linhas desta execução, em vez de recalculadas
    por caso.

    Nenhuma métrica levanta exceção por falta de dados: interseção vazia,
    pareamento sem correspondência ou caso sem nenhum bloco formado (volume
    inteiramente 0) resultam em np.nan nas métricas que dependem desses
    dados. Em particular, quando o volume de teste de um caso está
    inteiramente sem rótulo (ex.: kernel grande demais para a máscara, sem
    nenhum bloco formado), esse caso aparece na tabela com
    percentual_voxels_perdidos = 100 (voxels_perdidos() sobre o volume de
    teste) e com ari, concordancia_voxels_pct, deslocamento_centroide_
    espacial_medio/max e variacao_volume_max_pct em np.nan — ari e
    concordancia já resultam em nan naturalmente (interseção vazia);
    variacao_volume_max_pct é explicitamente forçado a nan neste caso, pois
    senão registraria uma queda de 100% em cada habitat de referência, um
    valor tecnicamente calculável mas sem sentido de comparação quando um
    dos lados não tem nenhum voxel rotulado.

    variacao_volume_max_pct é o maior |variacao_percentual| (valor absoluto)
    entre os habitats do caso, vindo de variacao_volume_habitats() — ou
    seja, a maior variação relativa de volume, para cima ou para baixo,
    entre os habitats daquele caso.

    Entradas:
        - volumes_ref: list[np.ndarray] — volumes de rótulos de referência,
          um por caso.
        - volumes_teste: list[np.ndarray] — volumes de rótulos da execução
          de teste, um por caso, na mesma ordem de volumes_ref.
        - mascaras: list[np.ndarray] — máscara original do tumor (antes do
          fechamento morfológico) de cada caso, na mesma ordem.
        - centros_ref: np.ndarray (k_ref, n_features) — cluster_centers_ da
          execução de referência.
        - centros_teste: np.ndarray (k_teste, n_features) — cluster_centers_
          da execução de teste.
        - metadados: dict com os identificadores da execução de teste (por
          exemplo {'parametro_variado': 'kernel', 'valor': 5}), replicado
          como colunas iniciais em todas as linhas da tabela.
        - spacing: tupla (dz, dy, dx), repassada a
          deslocamento_centroides_espacial() (None = distância em voxels).

    Saída:
        - pd.DataFrame com uma linha por caso e as colunas: as chaves de
          metadados (na ordem em que aparecem no dict), 'caso' (1-indexado),
          'ari', 'concordancia_voxels_pct', 'deslocamento_centroide_
          espacial_medio', 'deslocamento_centroide_espacial_max',
          'variacao_volume_max_pct', 'percentual_voxels_perdidos',
          'n_habitats_ref', 'n_habitats_teste', e por fim as três métricas
          por execução repetidas em todas as linhas: 'ari_conjunto',
          'deslocamento_centroide_features_medio',
          'deslocamento_centroide_features_max'.
    '''

    n_casos = len(volumes_ref)
    if len(volumes_teste) != n_casos or len(mascaras) != n_casos:
        raise ValueError(
            f"volumes_ref, volumes_teste e mascaras devem ter o mesmo número de casos "
            f"({len(volumes_ref)}, {len(volumes_teste)}, {len(mascaras)})."
        )

    # Pareamento GLOBAL, calculado uma única vez sobre todos os casos desta execução.
    pareamento = parear_rotulos_global(volumes_ref, volumes_teste)

    # Métricas POR EXECUÇÃO (não por caso), repetidas em todas as linhas.
    ari_conjunto_valor = ari_habitats_conjunto(volumes_ref, volumes_teste)
    deslocamentos_features = deslocamento_centroides_features(centros_ref, centros_teste, pareamento)
    deslocamento_features_medio, deslocamento_features_max = _media_e_maximo_sem_nan(deslocamentos_features.values())

    linhas = []
    for i in range(n_casos):
        volume_ref_caso = np.asarray(volumes_ref[i])
        volume_teste_caso = np.asarray(volumes_teste[i])
        mascara_caso = mascaras[i]

        # Caso degenerado: um dos lados não tem nenhum voxel rotulado (ex.: janela
        # grande demais não formou nenhum bloco). variacao_volume_max_pct precisa
        # de override explícito, pois não sairia nan "de graça" (ver docstring).
        sem_dados_ref = np.count_nonzero(volume_ref_caso) == 0
        sem_dados_teste = np.count_nonzero(volume_teste_caso) == 0

        ari = ari_habitats(volume_ref_caso, volume_teste_caso)

        concordancia_pct = concordancia_voxels(volume_ref_caso, volume_teste_caso, pareamento)
        if not np.isnan(concordancia_pct):
            concordancia_pct *= 100.0

        deslocamentos_espaciais = deslocamento_centroides_espacial(
            volume_ref_caso, volume_teste_caso, pareamento, spacing=spacing
        )
        deslocamento_espacial_medio, deslocamento_espacial_max = _media_e_maximo_sem_nan(deslocamentos_espaciais.values())

        if sem_dados_ref or sem_dados_teste:
            variacao_volume_max_pct = np.nan
        else:
            df_variacao = variacao_volume_habitats(volume_ref_caso, volume_teste_caso, pareamento)
            variacoes_validas = df_variacao['variacao_percentual'].dropna()
            variacao_volume_max_pct = float(variacoes_validas.abs().max()) if not variacoes_validas.empty else np.nan

        percentual_voxels_perdidos = voxels_perdidos(volume_teste_caso, mascara_caso)['percentual_perdido']

        n_habitats_ref = int(np.unique(volume_ref_caso[volume_ref_caso != 0]).size)
        n_habitats_teste = int(np.unique(volume_teste_caso[volume_teste_caso != 0]).size)

        linha = dict(metadados)
        linha.update({
            'caso': i + 1,
            'ari': ari,
            'concordancia_voxels_pct': concordancia_pct,
            'deslocamento_centroide_espacial_medio': deslocamento_espacial_medio,
            'deslocamento_centroide_espacial_max': deslocamento_espacial_max,
            'variacao_volume_max_pct': variacao_volume_max_pct,
            'percentual_voxels_perdidos': percentual_voxels_perdidos,
            'n_habitats_ref': n_habitats_ref,
            'n_habitats_teste': n_habitats_teste,
            'ari_conjunto': ari_conjunto_valor,
            'deslocamento_centroide_features_medio': deslocamento_features_medio,
            'deslocamento_centroide_features_max': deslocamento_features_max,
        })
        linhas.append(linha)

    colunas = list(metadados.keys()) + [
        'caso', 'ari', 'concordancia_voxels_pct',
        'deslocamento_centroide_espacial_medio', 'deslocamento_centroide_espacial_max',
        'variacao_volume_max_pct', 'percentual_voxels_perdidos',
        'n_habitats_ref', 'n_habitats_teste',
        'ari_conjunto', 'deslocamento_centroide_features_medio', 'deslocamento_centroide_features_max',
    ]

    return pd.DataFrame(linhas, columns=colunas)


def consolidar_tabelas(lista_dataframes: list, caminho_csv: str = None) -> pd.DataFrame:
    '''
    Concatena as tabelas de estabilidade (uma por execução de teste, no
    formato retornado por tabela_estabilidade_execucao()) em uma única
    tabela, e opcionalmente a exporta em CSV.

    Entradas:
        - lista_dataframes: list[pd.DataFrame] — uma tabela por execução de
          teste (mesmas colunas em todas, tipicamente vindas de
          tabela_estabilidade_execucao()).
        - caminho_csv: str = None — caminho onde salvar a tabela
          consolidada em CSV. Se None, nada é escrito em disco.

    Saída:
        - pd.DataFrame com todas as linhas concatenadas (índice
          reiniciado). Se lista_dataframes estiver vazia, retorna um
          pd.DataFrame vazio sem levantar exceção.
    '''

    if not lista_dataframes:
        tabela_consolidada = pd.DataFrame()
    else:
        tabela_consolidada = pd.concat(lista_dataframes, ignore_index=True)

    if caminho_csv is not None:
        tabela_consolidada.to_csv(caminho_csv, index=False)

    return tabela_consolidada


_PADRAO_CASO = re.compile(r'caso(\d+)', re.IGNORECASE)


def carregar_execucoes(caminho_saida) -> dict:
    '''
    Varre a pasta de saída do pipeline (ver pipeline_habitats_GridRegular_Pooled,
    em Habitats.py) e agrupa os arquivos de cada execução — um .json de
    parâmetros, um .npy de centroides e um .npy de volume de rótulos por caso —
    identificados por compartilharem o mesmo prefixo/sufixo no nome do arquivo
    (estrutura "plana": todos os arquivos direto em caminho_saida, nomeados
    como execucao_<config>.json, centroides_<config>.npy,
    rotulos_caso<n>_<config>.npy) ou por estarem juntos na mesma subpasta
    (estrutura "em subpasta": um .json, um .npy de centroides e os .npy de
    rótulos de uma execução dentro do mesmo diretório).

    A ordem dos volumes de rótulo dentro de cada execução é dada pelo número
    do caso extraído do nome do arquivo (rotulos_caso1_..., rotulos_caso2_...,
    ...), nunca pela ordem de leitura do diretório. É essa ordenação
    determinística — a mesma em todas as execuções — que garante que
    volumes_ref[i] e volumes_teste[i] sempre se refiram ao mesmo caso ao
    comparar duas execuções nas funções de métrica deste arquivo (ari_habitats,
    parear_rotulos_global, tabela_estabilidade_execucao, etc.).

    Entradas:
        - caminho_saida: str ou Path — pasta raiz de saída do pipeline,
          percorrida recursivamente.

    Saída:
        - dict {id_execucao: {"parametros": dict lido do .json, "volumes":
          lista de np.ndarray na ordem dos casos (1, 2, 3, 4, ...), "centros":
          np.ndarray dos cluster_centers_}}, uma entrada por execução
          completa encontrada. A chave especial '_ignoradas' contém a lista
          dos ids de execuções incompletas (faltando .json, centroides, ou
          nenhum .npy de rótulos reconhecível), descartadas sem levantar
          exceção.
    '''

    caminho_saida = Path(caminho_saida)
    execucoes = {}
    ignoradas = []

    for caminho_json in sorted(caminho_saida.rglob('*.json')):
        pasta = caminho_json.parent
        estrutura_plana = pasta.resolve() == caminho_saida.resolve()

        if estrutura_plana:
            # Arquivos da mesma execução compartilham um sufixo de configuração
            # no nome, dentro da própria pasta de saída (ex.: execucao_kernel3_..._k3.json,
            # centroides_kernel3_..._k3.npy, rotulos_caso1_kernel3_..._k3.npy).
            prefixo = caminho_json.stem
            if prefixo.startswith('execucao_'):
                prefixo = prefixo[len('execucao_'):]
            id_execucao = prefixo
            arquivo_centroides = pasta / f"centroides_{prefixo}.npy"
            arquivos_rotulos = sorted(pasta.glob(f"rotulos_caso*_{prefixo}.npy"))
        else:
            # Cada execução tem sua própria subpasta; os arquivos dentro dela
            # já pertencem, por construção, a uma única execução.
            id_execucao = str(pasta.relative_to(caminho_saida))
            candidatos_centroides = sorted(pasta.glob('centroides*.npy'))
            arquivo_centroides = candidatos_centroides[0] if candidatos_centroides else None
            arquivos_rotulos = sorted(pasta.glob('rotulos_caso*.npy'))

        arquivos_rotulos = [f for f in arquivos_rotulos if _PADRAO_CASO.search(f.stem)]

        if arquivo_centroides is None or not arquivo_centroides.exists() or not arquivos_rotulos:
            ignoradas.append(id_execucao)
            continue

        try:
            with open(caminho_json, 'r', encoding='utf-8') as arquivo:
                conteudo_json = json.load(arquivo)
            # O .json gravado pelo pipeline aninha os parâmetros da execução sob a
            # chave 'parametros' (junto de n_voxels_rotulados_por_caso e
            # n_blocos_por_caso); usa o conteúdo inteiro como fallback caso algum
            # .json não tenha essa chave.
            parametros = conteudo_json.get('parametros', conteudo_json)

            arquivos_rotulos_ordenados = sorted(
                arquivos_rotulos, key=lambda f: int(_PADRAO_CASO.search(f.stem).group(1))
            )
            volumes = [np.load(f) for f in arquivos_rotulos_ordenados]
            centros = np.load(arquivo_centroides)
        except (OSError, ValueError, json.JSONDecodeError):
            ignoradas.append(id_execucao)
            continue

        execucoes[id_execucao] = {
            'parametros': parametros,
            'volumes': volumes,
            'centros': centros,
        }

    execucoes['_ignoradas'] = ignoradas
    return execucoes


def identificar_referencia(execucoes: dict, parametros_ref: dict = None) -> str:
    '''
    Identifica, dentro do dict retornado por carregar_execucoes(), o id da
    execução de REFERÊNCIA — aquela cujos parâmetros no .json correspondem
    exatamente a parametros_ref.

    Essa execução é a que deve ser usada como volumes_ref/centros_ref nas
    funções de métrica deste arquivo (ari_habitats, parear_rotulos_global,
    tabela_estabilidade_execucao, etc.); as demais execuções entram como
    volumes_teste/centros_teste, uma de cada vez.

    Entradas:
        - execucoes: dict retornado por carregar_execucoes().
        - parametros_ref: dict = None — parâmetros que identificam a
          execução de referência. Se None, usa
          {'sigma': 0.4, 'kernel': 3, 'raio_fechamento': 1, 'passo': 1}.

    Saída:
        - str com o id_execucao (chave de `execucoes`) cujos parâmetros
          correspondem exatamente a parametros_ref.

    Levanta:
        - ValueError se nenhuma execução corresponder, listando os
          parâmetros de todas as execuções encontradas.
        - ValueError se mais de uma execução corresponder, indicando a
          ambiguidade e os ids envolvidos.
    '''

    if parametros_ref is None:
        parametros_ref = {'sigma': 0.4, 'kernel': 3, 'raio_fechamento': 1, 'passo': 1}

    execucoes_validas = {
        id_execucao: dados for id_execucao, dados in execucoes.items() if id_execucao != '_ignoradas'
    }

    candidatos = [
        id_execucao
        for id_execucao, dados in execucoes_validas.items()
        if all(dados['parametros'].get(chave) == valor for chave, valor in parametros_ref.items())
    ]

    if not candidatos:
        parametros_encontrados = {
            id_execucao: dados['parametros'] for id_execucao, dados in execucoes_validas.items()
        }
        raise ValueError(
            f"Nenhuma execução encontrada com os parâmetros de referência {parametros_ref}. "
            f"Parâmetros das execuções disponíveis: {parametros_encontrados}."
        )

    if len(candidatos) > 1:
        raise ValueError(
            f"Mais de uma execução corresponde aos parâmetros de referência {parametros_ref}: "
            f"{candidatos}. Ambiguidade não pode ser resolvida automaticamente."
        )

    return candidatos[0]
