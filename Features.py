import numpy as np
from scipy.stats import entropy

def features_primeira_ordem(supervoxel_valores: np.array):
    """
    Calcula as 17 features de primeira ordem de um conjunto de intensidades
    (um supervoxel).

    Entrada:
        - supervoxel_valores: array 1D com as intensidades dos voxels do supervoxel.
    Saída:
        - dict com as 17 features de primeira ordem.
    """

    supervoxel_valores = np.asarray(supervoxel_valores, dtype=np.float64)

    # Calculo das features com base nos valores de tons de cinza dos supervoxels.
    media = supervoxel_valores.mean()
    desvio_padrao = supervoxel_valores.std()
    mediana = np.median(supervoxel_valores)
    desvio = desvio_padrao if desvio_padrao != 0 else 1.0
    skewness = ((supervoxel_valores - media) ** 3).mean() / desvio ** 3
    kurtosis = ((supervoxel_valores - media) ** 4).mean() / desvio ** 4
    minimo = supervoxel_valores.min()
    maximo = supervoxel_valores.max()
    intervalo = maximo - minimo
    energia = (supervoxel_valores ** 2).sum()
    MAD_feature = np.abs(supervoxel_valores - media).mean()
    p10, p25, p75, p90 = np.percentile(supervoxel_valores, [10, 25, 75, 90])
    intervalo_interquartil = p75 - p25

    soma = supervoxel_valores.sum()
    if soma != 0:
        probabilidades = supervoxel_valores / soma
        # Remove probabilidades zero antes de calcular a entropia
        probabilidades_nonzero = probabilidades[probabilidades > 0]
        if len(probabilidades_nonzero) > 0:
            entropia = entropy(probabilidades_nonzero, base=2)
        else:
            entropia = 0.0
        Uniformidade = (probabilidades ** 2).sum()
    else:
        # Soma zero: todas as probabilidades são zero, entropia e uniformidade ficam 0
        entropia = 0.0
        Uniformidade = 0.0

    return {
        'media': media, 'desvio_padrao': desvio_padrao, 'mediana': mediana,
        'skewness': skewness, 'minimo': minimo, 'maximo': maximo,
        'range': intervalo, 'kurtosis': kurtosis, 'energia': energia,
        'MAD_feature': MAD_feature, 'p10': p10, 'p25': p25, 'p75': p75,
        'p90': p90, 'intervalo_interquartil': intervalo_interquartil,
        'entropia': entropia, 'Uniformidade': Uniformidade,
    }


def direcoes_3d():

  # 13 direções únicas em 3D (as opostas são equivalentes após a simetrização da GLCM).

  directions = []
  for dz in (-1, 0, 1):
    for dy in (-1, 0, 1):
      for dx in (-1, 0, 1):
        if (dz, dy, dx) == (0, 0, 0):
          continue
        if (dz, dy, dx) < (-dz, -dy, -dx):
          continue

        directions.append((dz, dy, dx))

  return directions


def extrair_glcm(grupo_voxels, n_levels=8):

    # GLCM isotrópica (soma das 13 direções) com quantização local por supervoxel.

    n_voxels = grupo_voxels.shape[0]

    vmin = grupo_voxels.min(axis=1, keepdims=True)
    vmax = grupo_voxels.max(axis=1, keepdims=True)
    rng = np.where((vmax - vmin) == 0, 1, vmax - vmin)
    quant = np.clip(np.floor((grupo_voxels - vmin) / rng * n_levels).astype(np.int32),
                    0, n_levels - 1).reshape(n_voxels, 3, 3, 3)

    #Essa é a parte em que os valores dos voxels são acessados, par por par, de acordo com o vetor de direções.
    glcm = np.zeros((n_voxels, n_levels, n_levels), dtype=np.float64)
    for dz, dy, dx in direcoes_3d():
        z1a, z1b = max(0, -dz), 3 - max(0, dz)
        y1a, y1b = max(0, -dy), 3 - max(0, dy)
        x1a, x1b = max(0, -dx), 3 - max(0, dx)
        z2a, z2b = max(0, dz), 3 - max(0, -dz)
        y2a, y2b = max(0, dy), 3 - max(0, -dy)
        x2a, x2b = max(0, dx), 3 - max(0, -dx)

        i = quant[:, z1a:z1b, y1a:y1b, x1a:x1b].reshape(n_voxels, -1)
        j = quant[:, z2a:z2b, y2a:y2b, x2a:x2b].reshape(n_voxels, -1)
        flat = i * n_levels + j
        rows = np.repeat(np.arange(n_voxels), flat.shape[1])
        np.add.at(glcm.reshape(n_voxels, -1), (rows, flat.ravel()), 1)

    glcm = glcm + glcm.transpose(0, 2, 1)
    totals = glcm.sum(axis=(1, 2), keepdims=True)
    P = glcm / np.where(totals == 0, 1, totals)

    i_idx = np.arange(n_levels).reshape(1, n_levels, 1)
    j_idx = np.arange(n_levels).reshape(1, 1, n_levels)
    diff = i_idx - j_idx

    # Estatísticas auxiliares da GLCM (médias, variâncias e covariância marginais),
    # usadas no cálculo das features abaixo.

    asm = (P ** 2).sum(axis=(1, 2))
    mu_i = (P * i_idx).sum(axis=(1, 2)).reshape(-1, 1, 1)
    mu_j = (P * j_idx).sum(axis=(1, 2)).reshape(-1, 1, 1)
    var_i = (P * (i_idx - mu_i) ** 2).sum(axis=(1, 2))
    var_j = (P * (j_idx - mu_j) ** 2).sum(axis=(1, 2))
    sigma = np.sqrt(var_i * var_j)
    cov = (P * (i_idx - mu_i) * (j_idx - mu_j)).sum(axis=(1, 2))

    mu_sum = (mu_i + mu_j).reshape(-1)                          # SumMean (Sum Average)
    cluster_tendency = (P * (i_idx + j_idx - mu_i - mu_j) ** 2).sum(axis=(1, 2))

    return np.column_stack([
        # asm: mede padrões homogêneos na imagem. Um valor maior implica mais pares
        # de intensidades vizinhas que se repetem com alta frequência (IBSI: ASM).
        asm,

        # entropia: mede a aleatoriedade/variabilidade das intensidades na vizinhança.
        -(P * np.log2(P + 1e-12)).sum(axis=(1, 2)),

        # contraste: mede a variação local de intensidade, favorecendo valores fora
        # da diagonal (i != j); valores maiores indicam maior disparidade entre voxels vizinhos.
        (P * diff ** 2).sum(axis=(1, 2)),

        # homogeneidade (ID): mede a homogeneidade local da imagem. Quanto mais
        # uniformes os níveis de cinza, menor o denominador e maior o valor final.
        (P / (1.0 + np.abs(diff))).sum(axis=(1, 2)),

        # correlacao: valor entre 0 (não correlacionado) e 1 (perfeitamente
        # correlacionado), mostra a dependência linear dos níveis de cinza na GLCM.
        cov / np.where(sigma == 0, 1, sigma),

        # variancia (Sum of Squares): mede a dispersão dos pares de intensidades
        # vizinhas em torno da intensidade média da GLCM (IBSI: Joint Variance).
        var_j,

        # sum_mean (Sum Average): mede a relação entre ocorrências de pares com
        # intensidades baixas e ocorrências de pares com intensidades altas.
        mu_sum,

        # probabilidade_maxima: ocorrência do par de intensidades vizinhas mais
        # predominante (IBSI: Joint maximum).
        P.max(axis=(1, 2)),

        # idm (Inverse Difference Moment): mede a homogeneidade local da imagem.
        # Os pesos do IDM são o inverso dos pesos do contraste, diminuindo
        # exponencialmente a partir da diagonal i=j na GLCM.
        (P / (1.0 + diff ** 2)).sum(axis=(1, 2)),

        # cluster_tendency: mede agrupamentos de voxels com níveis de cinza semelhantes.
        cluster_tendency,

        # dissimilaridade (Difference Average): mede a relação entre ocorrências de
        # pares com intensidades semelhantes e ocorrências de pares com intensidades diferentes.
        (P * np.abs(diff)).sum(axis=(1, 2)),
    ])


def features_textura(windows: np.ndarray, coordenadas: np.ndarray, n_levels: int = 8):
    """
    Calcula as 11 features de textura (GLCM) de um supervoxel, agregando por MÉDIA
    as texturas calculadas voxel-a-voxel (vizinhança 3x3x3).

    Entrada:
        - windows: np.ndarray com as vizinhanças 3x3x3 pré-calculadas de TODO o volume,
                   shape (Z, Y, X, 3, 3, 3), saída de sliding_window_view sobre o volume.
        - coordenadas: np.ndarray (n_voxels, 3) com os índices (z, y, x) dos voxels do
                       supervoxel, já no referencial de windows. Sobre um volume SEM
                       padding, windows[z-1, y-1, x-1] é a vizinhança centrada em
                       (z, y, x), então o chamador deve passar coordenadas - 1
                       (ver extrair(), em Habitats.py).
        - n_levels: int, número de níveis de quantização da GLCM.
    Saída:
        - dict com as 11 features de textura.
    """
    nomes_textura = ['asm', 'entropia', 'contraste', 'homogeneidade', 'correlacao',
                     'variancia', 'sum_mean', 'probabilidade_maxima', 'idm',
                     'cluster_tendency', 'dissimilaridade']

    vizinhancas = windows[coordenadas[:, 0],
                          coordenadas[:, 1],
                          coordenadas[:, 2]].reshape(len(coordenadas), 27)

    texturas = extrair_glcm(vizinhancas, n_levels=n_levels)   # (n_voxels, 11)
    texturas_supervoxel = texturas.mean(axis=0)               # agregação por média -> (11,)

    return dict(zip(nomes_textura, texturas_supervoxel))
