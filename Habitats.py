#Bibliotecas
import os
import json
import SimpleITK as sitk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.ndimage import gaussian_filter, binary_closing
import matplotlib.cm as cm
from Features import features_primeira_ordem, features_textura

#=====================================================================================================================
#=====================================================================================================================

SPACING_PADRAO = 0.7


#===============================================================================
# REAMOSTRAGEM PARA ESPAÇAMENTO ISOTRÓPICO
#===============================================================================

def isotropico(imagem_sitk, novo_spacing, interpolador, default_value=0):

    '''
    Reamostra uma imagem SimpleITK para espaçamento isotrópico.

    Em RM o voxel costuma ser anisotrópico (alta resolução no plano, fatia grossa
    em z). Isso torna o bloco KxKxK e a GLCM isotrópica da etapa de extração
    fisicamente inconsistentes: 1 voxel em z cobre muito mais mm do que no plano.
    Reamostrar para voxel cúbico corrige isso na origem.

    ATENÇÃO ao interpolador:
        - imagem (intensidades): sitk.sitkLinear (ou sitk.sitkBSpline).
        - máscara/segmentação  : sitk.sitkNearestNeighbor, para permanecer binária.
          Interpolação linear numa máscara gera valores fracionários e, como o
          restante do pipeline faz (seg > 0), a máscara "incharia".

    Entradas:
        - imagem_sitk : sitk.Image a reamostrar.
        - novo_spacing: float — espaçamento isotrópico de destino, em mm (x=y=z).
        - interpolador: constante de interpolação do SimpleITK.
        - default_value: valor para voxels fora do campo original (fundo = 0).

    Saída:
        - sitk.Image reamostrada, com origin e direction preservados.

    Obs: SimpleITK trabalha em (x, y, z) — diferente do array numpy (z, y, x).
    '''

    spacing_orig = imagem_sitk.GetSpacing()   # (x, y, z), em mm
    size_orig = imagem_sitk.GetSize()         # (x, y, z), em voxels
    novo_spacing_vec = [float(novo_spacing)] * 3

    # Novo número de voxels por eixo para cobrir a mesma extensão física.
    novo_size = [
        int(round(size_orig[d] * spacing_orig[d] / novo_spacing_vec[d]))
        for d in range(3)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(novo_spacing_vec)
    resampler.SetSize(novo_size)
    resampler.SetOutputOrigin(imagem_sitk.GetOrigin())
    resampler.SetOutputDirection(imagem_sitk.GetDirection())
    resampler.SetInterpolator(interpolador)
    resampler.SetDefaultPixelValue(default_value)

    return resampler.Execute(imagem_sitk)

def ler_caso_isotropico(caminho_imagem, caminho_mascara, novo_spacing=SPACING_PADRAO):

    '''
    Lê o exame e a máscara de um caso e os devolve como arrays numpy (z, y, x)
    JÁ REAMOSTRADOS para espaçamento isotrópico, com a mesma lógica usada em
    organizar.py (função isotropico + reamostragem da máscara sobre a grade da
    imagem).

    Motivo: o voxel de RM costuma ser anisotrópico (alta resolução no plano,
    fatia grossa em z). Sem reamostrar, o bloco KxKxK da segmentação e a GLCM
    isotrópica de Features.py representam volumes físicos diferentes em cada
    eixo e em cada paciente.

    Interpoladores (iguais aos de organizar.py):
        - imagem (intensidades): sitk.sitkLinear.
        - máscara (segmentação) : sitk.sitkNearestNeighbor, sobre a MESMA grade
          da imagem reamostrada (SetReferenceImage), o que garante alinhamento
          voxel a voxel e mantém a máscara binária.

    Entradas:
        - caminho_imagem: str — caminho do exame.
        - caminho_mascara: str — caminho da segmentação.
        - novo_spacing: float — espaçamento isotrópico de destino, em mm, igual
          para todos os casos. Se None, NÃO reamostra (comportamento anterior
          a esta função).

    Saídas:
        - imagem: np.ndarray (z, y, x) com as intensidades.
        - mascara: np.ndarray (z, y, x) com a segmentação.
    '''

    imagem_sitk = sitk.ReadImage(caminho_imagem)
    mascara_sitk = sitk.ReadImage(caminho_mascara)

    if novo_spacing is not None:
        # Imagem (intensidades): interpolação LINEAR.
        imagem_sitk = isotropico(imagem_sitk, float(novo_spacing), sitk.sitkLinear)

        # Máscara: VIZINHO MAIS PRÓXIMO, sobre a mesma grade da imagem reamostrada.
        resampler_mascara = sitk.ResampleImageFilter()
        resampler_mascara.SetReferenceImage(imagem_sitk)
        resampler_mascara.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler_mascara.SetDefaultPixelValue(0)
        mascara_sitk = resampler_mascara.Execute(mascara_sitk)

    return sitk.GetArrayFromImage(imagem_sitk), sitk.GetArrayFromImage(mascara_sitk)

#===============================================================================
# NORMALIZAÇÃO DE INTENSIDADES PARA [0, 255]
#===============================================================================

def normalizar_intensidades(imagem, mascara, n_sigma=3):

    '''
    Normaliza as intensidades de uma imagem para o intervalo [0, 255], usando
    como referência a média e o desvio padrão dos voxels do tumor.

    Entradas:
        - imagem  : np.ndarray 3D com as intensidades originais do exame.
        - mascara : np.ndarray 3D binária do tumor (mesma shape de imagem).
        - n_sigma : float = 3 — número de desvios padrão usado para definir
          os limites de recorte.

    Saída:
        - np.ndarray 3D, dtype float64, com intensidades no intervalo [0, 255].
    '''

    voxels_referencia = imagem[mascara > 0]
    if voxels_referencia.size == 0:
        raise ValueError("A máscara não contém nenhum voxel; não é possível "
                         "calcular média e desvio padrão de referência.")

    media = voxels_referencia.mean()
    desvio_padrao = voxels_referencia.std(ddof=0)

    if desvio_padrao == 0:
        return np.zeros_like(imagem, dtype=np.float64)

    limite_inferior = media - n_sigma * desvio_padrao
    limite_superior = media + n_sigma * desvio_padrao

    imagem_recortada = np.clip(imagem, limite_inferior, limite_superior)
    imagem_normalizada = (imagem_recortada - limite_inferior) / (limite_superior - limite_inferior) * 255

    return imagem_normalizada.astype(np.float64)

#=====================================================================================================================
#=====================================================================================================================

def segmentacao(caminho_imagem, caminho_mascara, kernel: int = 3, passo: int = 1, novo_spacing=SPACING_PADRAO):

    '''
        Esta função tem como principal objetivo a obtenção de um array contendo a segmentação
    da imagem, utilizando um grid regular de KxKxK.

    Entradas:
        - caminho_imagem: str
        - caminho_mascara: str
        - kernel: int = 3 — tamanho da janela (3, 5, 7, 9, 11).
        - passo: int = 1 — passo da janela deslizante. passo < kernel gera sobreposição.
        - novo_spacing: float — espaçamento isotrópico (mm) da reamostragem
          aplicada na leitura (ver ler_caso_isotropico). Precisa ser o MESMO
          usado em extrair() e na visualização, senão os shapes divergem.

    Saídas:
        - segmentacao_sem_slic: np.ndarray com os labels dos supervoxels.
        - supervoxels_sem_slic: dict com as coordenadas de cada supervoxel.

    '''

    imagem, mascara = ler_caso_isotropico(caminho_imagem, caminho_mascara, novo_spacing=novo_spacing)

    ID_supervoxels = 1
    segmentacao_kernel = np.zeros(imagem.shape, dtype=np.int32)
    supervoxels_kernel = {}

    for z in range(0, imagem.shape[0] - kernel + 1, passo):
        for y in range(0, imagem.shape[1] - kernel + 1, passo):
            for x in range(0, imagem.shape[2] - kernel + 1, passo):

                z_final = z + kernel
                y_final = y + kernel
                x_final = x + kernel

                mascara_atual = mascara[z:z_final, y:y_final, x:x_final]

                # Máscara apenas com valores > 0
                if np.all(mascara_atual > 0):
                    segmentacao_kernel[z:z_final, y:y_final, x:x_final] = ID_supervoxels

                    # Gerar as coordenadas
                    coordenadas = []
                    for sz in range(z, z_final):
                        for sy in range(y, y_final):
                            for sx in range(x, x_final):
                                coordenadas.append((sz, sy, sx))

                    # Armazenar as coordenadas
                    supervoxels_kernel[ID_supervoxels] = coordenadas
                    ID_supervoxels += 1

    return segmentacao_kernel, supervoxels_kernel

#=====================================================================================================================
#=====================================================================================================================
def fechar_esfera(volume, raio, iterations):

    r = int(raio)
    d = 2 * r + 1
    centro = np.array([r, r, r])
    coords = np.indices((d, d, d)).T.reshape(-1, 3)
    mask = np.linalg.norm(coords - centro, axis=1) <= raio
    struct = mask.reshape(d, d, d)
    closed = binary_closing(volume, structure=struct, iterations = iterations)

    return closed

#=====================================================================================================================
#=====================================================================================================================

def extrair(caminho_imagem, caminho_mascara, supervoxels_kernel, kernel: int = 3, sigma: float = 0.2, raio_fechamento: int = 3, iterations: int = 1, novo_spacing=SPACING_PADRAO, normalizar: bool = True, n_sigma: float = 3):

    '''
    Entradas adicionais:
        - normalizar: bool = True — se True, normaliza as intensidades para
          [0, 255] via normalizar_intensidades(), usando a máscara ORIGINAL
          (antes do fechamento morfológico), logo após a leitura isotrópica e
          antes do gaussian_filter. Isso mantém a normalização independente de
          raio_fechamento, para não confundir os dois efeitos nos testes de
          sensibilidade.
        - n_sigma: float = 3 — número de desvios padrão usado por
          normalizar_intensidades() para definir os limites de recorte.
    '''

    # A leitura é feita já reamostrada para espaçamento isotrópico (ver
    # ler_caso_isotropico). O filtro gaussiano e o fechamento morfológico
    # continuam sendo aplicados DEPOIS, agora sobre uma grade de voxel cúbico —
    # sigma e raio_fechamento passam a ter o mesmo significado físico nos 3 eixos.
    imagem, mascara = ler_caso_isotropico(caminho_imagem, caminho_mascara, novo_spacing=novo_spacing)

    if normalizar:
        imagem = normalizar_intensidades(imagem, mascara, n_sigma=n_sigma)

    imagem = gaussian_filter(imagem, sigma=sigma)
    mascara_fechada = fechar_esfera(mascara, raio_fechamento, iterations)

    imagem_mascarada = imagem * mascara_fechada
    windows = np.lib.stride_tricks.sliding_window_view(imagem_mascarada.astype(np.float64), (3, 3, 3))

    Dz, Dy, Dx = imagem.shape
    n_voxels_esperados = kernel ** 3
    supervoxel_features_list = []

    for label, coordenadas_list in supervoxels_kernel.items():
        coordenadas = np.array(coordenadas_list)          # (n_voxels, 3) em ordem (z, y, x)

        # Filtrar voxels de borda (sem vizinhança 3x3x3 completa no array sem padding)
        interior = (
            (coordenadas[:, 0] >= 1) & (coordenadas[:, 0] <= Dz - 2) &
            (coordenadas[:, 1] >= 1) & (coordenadas[:, 1] <= Dy - 2) &
            (coordenadas[:, 2] >= 1) & (coordenadas[:, 2] <= Dx - 2)
        )
        coordenadas = coordenadas[interior]

        if len(coordenadas) < n_voxels_esperados:
            continue

        zs, ys, xs = coordenadas[:, 0], coordenadas[:, 1], coordenadas[:, 2]
        voxel_intensities = imagem_mascarada[zs, ys, xs]

        # 17 features de primeira ordem + 11 features de textura (GLCM)
        # Sem padding: windows[z-1, y-1, x-1] é a vizinhança centrada em (z, y, x)
        features_dict = {'Supervoxel_ID': label}
        features_dict.update(features_primeira_ordem(voxel_intensities))
        features_dict.update(features_textura(windows, coordenadas - 1))
        supervoxel_features_list.append(features_dict)

    if supervoxel_features_list:
        radiomic_features_df = pd.DataFrame(supervoxel_features_list)
        print(f"Extração de features concluída para {radiomic_features_df.shape[0]} supervoxels.")
        return radiomic_features_df

    print("Nenhuma feature foi extraída. Verifique a geração dos supervoxels.")
    return pd.DataFrame()

#=====================================================================================================================
#=====================================================================================================================

def clusterizar(radiomic_features_df, supervoxels_sem_slic, k = 3):

    '''
    Esta função tem como principal objetivo a clusterização dos dados, considerando os 
    valores das features extraídas e os supervoxels que foram formados. 

    Entredas: 
        - radiomic_features_df
        - supervoxels_sem_slic

    Saídas:
        - final_df
        - cluster_map
        - num_clusters

    '''

    # Check if radiomic_features_df is not empty before proceeding with clustering
    if not radiomic_features_df.empty:
        X = radiomic_features_df.drop(columns=['Supervoxel_ID'])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10) # n_init para evitar warnings
        labels_originais = kmeans.fit_predict(X_scaled)

        media_por_cluster = radiomic_features_df.groupby(labels_originais)['media'].mean()
        ordem = media_por_cluster.sort_values().index
        mapa_reordenado = {old: new for new, old in enumerate(ordem)}
        radiomic_features_df['Cluster_Label'] = np.array([mapa_reordenado[l] for l in labels_originais])

        print(f"K-means aplicado com {k} clusters.")
        print("Distribuição dos clusters:")
        print(radiomic_features_df['Cluster_Label'].value_counts())

        cluster_map = radiomic_features_df.set_index('Supervoxel_ID')['Cluster_Label'].to_dict()
        num_clusters = radiomic_features_df['Cluster_Label'].nunique()

        supervoxel_coords_df = pd.DataFrame({
            'Supervoxel_ID': list(supervoxels_sem_slic.keys()),
            'Voxel_Coordinates': list(supervoxels_sem_slic.values())
        })

        final_df = pd.merge(
            radiomic_features_df[['Supervoxel_ID', 'Cluster_Label']],
            supervoxel_coords_df,
            on='Supervoxel_ID',
            how='left'
        )
        print("\nDataFrame final com Supervoxel_ID, Cluster_Label e Voxel_Coordinates (primeiras 5 linhas):")
        print(final_df.head().to_string())
    else:
        print("Não há dados para aplicar K-means. O DataFrame de features está vazio.")
        final_df = pd.DataFrame()
        cluster_map = {}
        num_clusters = 0

    return final_df, cluster_map, num_clusters

#=====================================================================================================================
#=====================================================================================================================

def clusterizar_pooled(lista_features_df, lista_supervoxels, k=3, caminho_centroides='centroides_pooled.npy'):

    '''
    Versão conjunta (pooled) de clusterizar(): concatena as features de vários
    casos e ajusta o StandardScaler e o K-means UMA ÚNICA VEZ sobre a matriz
    concatenada, depois redistribui os rótulos resultantes (rótulo bruto do
    K-means, sem reordenação) de volta para cada caso. Como o ajuste é único
    sobre os casos concatenados, os rótulos já são comparáveis entre eles.

    Entradas:
        - lista_features_df: list[pd.DataFrame] — uma tabela de features (saída
          de extrair()) por caso, na mesma ordem dos casos.
        - lista_supervoxels: list[dict] — supervoxels_kernel de cada caso, na
          mesma ordem de lista_features_df.
        - k: int — número de clusters do K-means.
        - caminho_centroides: str — caminho (.npy) onde salvar os centroides
          do ajuste conjunto.

    Saídas:
        - lista_resultados: list[tuple(final_df, cluster_map, num_clusters)],
          um por caso, no mesmo formato retornado por clusterizar().
    '''

    if not any(not df.empty for df in lista_features_df):
        print("Não há dados para aplicar K-means. Todos os DataFrames de features estão vazios.")
        return [(pd.DataFrame(), {}, 0) for _ in lista_features_df]

    # Passo 1/2: concatenar as features de todos os casos, guardando os deslocamentos
    # de índice (offsets) necessários para desfazer a concatenação depois.
    tamanhos = [len(df) for df in lista_features_df]
    offsets = np.cumsum([0] + tamanhos)

    features_concatenadas = pd.concat(lista_features_df, ignore_index=True)
    X = features_concatenadas.drop(columns=['Supervoxel_ID'])

    # Passo 3: StandardScaler e K-means ajustados UMA ÚNICA VEZ sobre a matriz conjunta.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_kmeans = kmeans.fit_predict(X_scaled)
    features_concatenadas['Cluster_Label'] = labels_kmeans

    # Passo 4: centroides do ajuste conjunto, salvos em disco.
    np.save(caminho_centroides, kmeans.cluster_centers_)

    print(f"K-means aplicado com {k} clusters (ajuste conjunto sobre {len(lista_features_df)} casos).")
    print(f"Centroides do ajuste conjunto salvos em '{caminho_centroides}'.")
    print("Distribuição dos clusters (conjunto):")
    print(features_concatenadas['Cluster_Label'].value_counts())

    num_clusters_total = features_concatenadas['Cluster_Label'].nunique()

    # Passo 5: fatiar os rótulos de volta por caso, usando os offsets do passo 2.
    lista_resultados = []
    for i, (df_original, supervoxels_caso) in enumerate(zip(lista_features_df, lista_supervoxels)):
        if df_original.empty:
            lista_resultados.append((pd.DataFrame(), {}, 0))
            continue

        inicio, fim = offsets[i], offsets[i + 1]
        df_caso = df_original.copy()
        df_caso['Cluster_Label'] = labels_kmeans[inicio:fim]

        print(f"\nCaso {i + 1}: distribuição dos clusters:")
        print(df_caso['Cluster_Label'].value_counts())

        cluster_map = df_caso.set_index('Supervoxel_ID')['Cluster_Label'].to_dict()

        supervoxel_coords_df = pd.DataFrame({
            'Supervoxel_ID': list(supervoxels_caso.keys()),
            'Voxel_Coordinates': list(supervoxels_caso.values())
        })

        final_df = pd.merge(
            df_caso[['Supervoxel_ID', 'Cluster_Label']],
            supervoxel_coords_df,
            on='Supervoxel_ID',
            how='left'
        )

        lista_resultados.append((final_df, cluster_map, num_clusters_total))

    return lista_resultados

#=====================================================================================================================
#=====================================================================================================================

def plot_clustered_supervoxels_overlay(original_volume, segments_volume, cluster_map, num_clusters, n_cols=4, alpha=0.6, blur_sigma=0.5, caminho_pdf=None, titulo=None):
    
    """
    Plota os supervoxels clusterizados sobre as imagens originais, com borramento e transparência.

    Args:
        original_volume (np.array): O volume da imagem original (exame_array).
        segments_volume (np.array): O volume dos labels dos superpixels (segments_slic_masked).
        cluster_map (dict): Um dicionário mapeando Supervoxel_ID para Cluster_Label.
        num_clusters (int): O número total de clusters.
        n_cols (int): Número de colunas para o grid de plots.
        alpha (float): Transparência do overlay dos clusters (0.0 a 1.0).
        blur_sigma (float): Sigma para o filtro Gaussiano de borramento nos clusters.
        caminho_pdf (str | None): se informado, a figura é salva em PDF nesse caminho
            e fechada, sem ser exibida. Se None, a figura é exibida (plt.show()).
        titulo (str | None): título geral da figura (caso e parâmetros da execução).
    """
    n_slices = original_volume.shape[0]
    n_rows = (n_slices + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5))

    if n_rows * n_cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    for i in range(n_slices):
        ax = axes[i]
        z_slice = i

        original_slice = original_volume[z_slice, :, :]
        segments_slice = segments_volume[z_slice, :, :]

        colored_overlay_canvas = np.zeros((*original_slice.shape, 3), dtype=float)

        unique_superpixels_in_slice = np.unique(segments_slice)
        for s_id in unique_superpixels_in_slice:
            if s_id != 0 and s_id in cluster_map:
                cluster_idx = cluster_map[s_id]
                color = cm.tab10(cluster_idx)[:3]
                colored_overlay_canvas[segments_slice == s_id] = color

        if blur_sigma > 0:
            blurred_colored_overlay_canvas = gaussian_filter(colored_overlay_canvas, sigma=(blur_sigma, blur_sigma, 0))
        else:
            blurred_colored_overlay_canvas = colored_overlay_canvas

        max_val = original_slice.max()
        original_slice_normalized = np.clip(original_slice / max_val if max_val > 0 else original_slice, 0, 1)
        original_slice_rgb = cm.gray(original_slice_normalized)[..., :3]

        final_display_slice = (original_slice_rgb * (1 - alpha)) + (blurred_colored_overlay_canvas * alpha)

        ax.imshow(final_display_slice)
        ax.set_title(f'Habitat Z={z_slice} (Cluster)')
        ax.axis('off')

    for j in range(n_slices, len(axes)):
        fig.delaxes(axes[j])

    if titulo is not None:
        fig.suptitle(titulo, fontsize=14)
        plt.tight_layout(rect=(0, 0, 1, 0.97))
    else:
        plt.tight_layout()

    if caminho_pdf is not None:
        pasta_pdf = os.path.dirname(caminho_pdf)
        if pasta_pdf:
            os.makedirs(pasta_pdf, exist_ok=True)
        fig.savefig(caminho_pdf, format='pdf', bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

#=====================================================================================================================
#=====================================================================================================================

def pipeline_habitats_GridRegular(caminho_imagem: str, caminho_mascara: str, sigma: float = 0.2, kernel: int = 3, raio_fechamento: int = 3, passo: int = 1, iterations: int = 1, novo_spacing=SPACING_PADRAO):

    '''
    Função que roda em conjunto e em ordem todas as funções referentes à segmentação em
    habitats usando um grid regular.

    Entrada:
        - caminho_imagem: str
        - caminho_mascara: str
        - sigma: float = 0.2 — sigma do filtro gaussiano.
        - kernel: int = 3 — tamanho da janela (3, 5, 7, 9, 11).
        - raio_fechamento: int = 3 — raio do kernel esférico de fechamento.
        - passo: int = 1 — passo da janela deslizante.
        - novo_spacing: float — espaçamento isotrópico (mm) usado na leitura das
          imagens, igual para todos os casos (ver ler_caso_isotropico). Use None
          para não reamostrar. Valores menores aumentam o número de voxels com o
          CUBO do inverso do spacing.

    Saída:
        - None

    '''

    # PRIMEIRA ETAPA - SEGMENTAÇÃO
    print(80 * "=")
    print(f"Etapa de segmentação Iniciada (kernel={kernel}x{kernel}x{kernel}, spacing={novo_spacing} mm)")
    segmentacao_kernel, supervoxels_kernel = segmentacao(caminho_imagem, caminho_mascara, kernel=kernel, passo=passo, novo_spacing=novo_spacing)
    print("Etapa de segmentação Finalizada")
    print(80 * "=")

    #SEGUNDA ETAPA - EXTRAÇÃO DAS FEATURES
    print(80 * "=")
    print("Etapa de extração de features Iniciada")
    radiomic_features_df = extrair(caminho_imagem, caminho_mascara, supervoxels_kernel, kernel=kernel, sigma=sigma, raio_fechamento=raio_fechamento, iterations = iterations, novo_spacing=novo_spacing)
    print("Etapa de extração de features Finalizada")
    print(80 * "=")

    #TERCEIRA ETAPA - CLUSTERIZAÇÃO
    print(80 * "=")
    print("Etapa de clusterização Iniciada")
    final_df, cluster_map, num_clusters = clusterizar(radiomic_features_df, supervoxels_kernel)
    print("Etapa de clusterização Finalizada")
    print(80 * "=")

    #QUARTA ETAPA - VISUALIZAÇÃO HABITATS
    if num_clusters > 0:
        # A imagem de fundo precisa vir da MESMA grade reamostrada usada na
        # segmentação, senão o shape não bate com segmentacao_kernel.
        imagem, _ = ler_caso_isotropico(caminho_imagem, caminho_mascara, novo_spacing=novo_spacing)
        plot_clustered_supervoxels_overlay(imagem, segmentacao_kernel, cluster_map, num_clusters, n_cols=5, alpha=0.5, blur_sigma=0)
    else:
        print("Não há clusters para visualizar, pois nenhum supervoxel foi processado com features e clusterização.")

#=====================================================================================================================
#=====================================================================================================================

def construir_volume_rotulos(segmentacao_kernel, cluster_map, dtype=np.int16):

    '''
    Constrói o volume de rótulos por voxel de um caso, a partir do volume de
    IDs de supervoxel (segmentacao_kernel) e do cluster_map daquele caso
    (Supervoxel_ID -> Cluster_Label).

    Convenção:
        - 0: voxel sem rótulo (fora da máscara, supervoxel descartado em
          extrair() por estar na borda ou incompleto, ou não coberto por
          nenhum bloco).
        - 1..k: Cluster_Label + 1.

    Entradas:
        - segmentacao_kernel: np.ndarray com o ID do supervoxel em cada voxel
          (0 = fora), saída de segmentacao().
        - cluster_map: dict Supervoxel_ID -> Cluster_Label (inteiros 0..k-1)
          do caso correspondente.
        - dtype: dtype inteiro do volume de saída.

    Saída:
        - volume_rotulos: np.ndarray com o mesmo shape de segmentacao_kernel.
    '''

    lut = np.zeros(int(segmentacao_kernel.max()) + 1, dtype=dtype)
    for supervoxel_id, cluster_label in cluster_map.items():
        lut[supervoxel_id] = cluster_label + 1

    return lut[segmentacao_kernel]

#=====================================================================================================================
#=====================================================================================================================

def pipeline_habitats_GridRegular_Pooled(caminhos_imagens: list, caminhos_mascaras: list, sigma: float = 0.2, kernel: int = 3, raio_fechamento: int = 3, passo: int = 1, iterations: int = 1, k: int = 3, caminho_centroides: str = 'centroides_pooled.npy', caminho_saida: str = None, novo_spacing=SPACING_PADRAO, pasta_visualizacoes: str = None, normalizar: bool = True, n_sigma: float = 3):

    '''
    Versão conjunta (pooled) de pipeline_habitats_GridRegular(): roda a
    segmentação e a extração de features separadamente para cada caso, mas
    ajusta o K-means UMA ÚNICA VEZ sobre a concatenação das features de todos
    os casos (via clusterizar_pooled), tornando os rótulos de cluster
    comparáveis entre os casos. Ao final, gera a visualização dos habitats de
    cada caso.

    Entrada:
        - caminhos_imagens: list[str] — caminho do exame de cada caso.
        - caminhos_mascaras: list[str] — caminho da máscara de cada caso, na
          mesma ordem de caminhos_imagens.
        - sigma, kernel, raio_fechamento, passo, iterations: iguais aos de
          pipeline_habitats_GridRegular(), aplicados a cada caso.
        - k: int = 3 — número de clusters do K-means (ajuste conjunto).
        - caminho_centroides: str — caminho (.npy) onde salvar os centroides
          do ajuste conjunto.
        - caminho_saida: str = None — pasta onde salvar os artefatos desta
          execução (um .npy por caso com o volume de rótulos por voxel, via
          construir_volume_rotulos(), e um .json com os parâmetros e
          estatísticas da execução). Quando None, nada é salvo em disco e o
          comportamento é o mesmo de antes desta opção existir.
        - novo_spacing: float — espaçamento isotrópico (mm) usado na leitura das
          imagens (ver ler_caso_isotropico), FIXO e igual para todos os casos.
          Isso é o que torna o bloco KxKxK e as features de textura comparáveis
          entre os casos na clusterização conjunta. Use None para não reamostrar.
        - normalizar: bool = True — repassado a extrair(): se True, normaliza
          as intensidades de cada caso para [0, 255] via normalizar_intensidades()
          antes da extração de features.
        - n_sigma: float = 3 — repassado a extrair(): número de desvios padrão
          usado por normalizar_intensidades() para definir os limites de recorte.

    Saída:
        - None

    '''

    if len(caminhos_imagens) != len(caminhos_mascaras):
        raise ValueError("caminhos_imagens e caminhos_mascaras devem ter o mesmo tamanho.")

    n_casos = len(caminhos_imagens)
    lista_features_df = []
    lista_supervoxels = []
    lista_segmentacoes = []

    config_id = (f"kernel{kernel}_sigma{sigma}_raio{raio_fechamento}_passo{passo}"
                 f"_iter{iterations}_k{k}_sp{novo_spacing}_norm{int(normalizar)}"
                 f"_ns{n_sigma}")
    if caminho_saida is not None:
        os.makedirs(caminho_saida, exist_ok=True)
        caminho_centroides = os.path.join(caminho_saida, f"centroides_{config_id}.npy")

    # PRIMEIRA E SEGUNDA ETAPAS - SEGMENTAÇÃO E EXTRAÇÃO (por caso)
    for i, (caminho_imagem, caminho_mascara) in enumerate(zip(caminhos_imagens, caminhos_mascaras)):
        print(80 * "=")
        print(f"Caso {i + 1}/{n_casos} — Etapa de segmentação Iniciada (kernel={kernel}x{kernel}x{kernel}, spacing={novo_spacing} mm)")
        segmentacao_kernel, supervoxels_kernel = segmentacao(caminho_imagem, caminho_mascara, kernel=kernel, passo=passo, novo_spacing=novo_spacing)
        print(f"Caso {i + 1}/{n_casos} — Etapa de segmentação Finalizada")

        print(f"Caso {i + 1}/{n_casos} — Etapa de extração de features Iniciada")
        radiomic_features_df = extrair(caminho_imagem, caminho_mascara, supervoxels_kernel, kernel=kernel, sigma=sigma, raio_fechamento=raio_fechamento, iterations=iterations, novo_spacing=novo_spacing, normalizar=normalizar, n_sigma=n_sigma)
        print(f"Caso {i + 1}/{n_casos} — Etapa de extração de features Finalizada")
        print(80 * "=")

        lista_features_df.append(radiomic_features_df)
        lista_supervoxels.append(supervoxels_kernel)
        lista_segmentacoes.append(segmentacao_kernel)

    #TERCEIRA ETAPA - CLUSTERIZAÇÃO CONJUNTA (POOLED)
    print(80 * "=")
    print("Etapa de clusterização conjunta (pooled) Iniciada")
    resultados = clusterizar_pooled(lista_features_df, lista_supervoxels, k=k, caminho_centroides=caminho_centroides)
    print("Etapa de clusterização conjunta (pooled) Finalizada")
    print(80 * "=")

    #QUARTA ETAPA - VISUALIZAÇÃO HABITATS (por caso)
    n_voxels_rotulados_por_caso = []
    for i, caminho_imagem in enumerate(caminhos_imagens):
        final_df, cluster_map, num_clusters = resultados[i]

        if caminho_saida is not None:
            volume_rotulos = construir_volume_rotulos(lista_segmentacoes[i], cluster_map, dtype=np.int16)
            caminho_npy = os.path.join(caminho_saida, f"rotulos_caso{i + 1}_{config_id}.npy")
            np.save(caminho_npy, volume_rotulos)
            n_voxels_rotulados_por_caso.append(int(np.count_nonzero(volume_rotulos)))

        if num_clusters > 0:
            # Mesma grade reamostrada usada na segmentação deste caso.
            imagem, _ = ler_caso_isotropico(caminho_imagem, caminhos_mascaras[i], novo_spacing=novo_spacing)
            print(f"Caso {i + 1}/{n_casos} — gerando visualização dos habitats")

            caminho_pdf = None
            if pasta_visualizacoes is not None:
                # Uma subpasta por execução, com o mesmo nome usado em caminho_saida,
                # contendo um PDF por caso.
                nome_execucao = os.path.basename(os.path.normpath(caminho_saida)) if caminho_saida else config_id
                pasta_execucao = os.path.join(pasta_visualizacoes, nome_execucao)
                os.makedirs(pasta_execucao, exist_ok=True)
                caminho_pdf = os.path.join(pasta_execucao, f"caso{i + 1}_{config_id}.pdf")

            titulo = (f"Caso {i + 1} — sigma={sigma}, kernel={kernel}, "
                      f"raio={raio_fechamento}, passo={passo}, k={k}, spacing={novo_spacing} mm")

            plot_clustered_supervoxels_overlay(imagem, lista_segmentacoes[i], cluster_map, num_clusters,
                                              n_cols=5, alpha=0.5, blur_sigma=0,
                                              caminho_pdf=caminho_pdf, titulo=titulo)
            if caminho_pdf is not None:
                print(f"    PDF salvo em: {caminho_pdf}")
        else:
            print(f"Caso {i + 1}/{n_casos}: não há clusters para visualizar, pois nenhum supervoxel foi processado com features e clusterização.")

    #QUINTA ETAPA - METADADOS DA EXECUÇÃO
    if caminho_saida is not None:
        info_execucao = {
            'parametros': {
                'sigma': sigma, 'kernel': kernel, 'raio_fechamento': raio_fechamento,
                'passo': passo, 'iterations': iterations, 'k': k,
                'novo_spacing': novo_spacing, 'normalizar': normalizar, 'n_sigma': n_sigma,
            },
            'n_voxels_rotulados_por_caso': n_voxels_rotulados_por_caso,
            'n_blocos_por_caso': [len(df) for df in lista_features_df],
        }
        caminho_json = os.path.join(caminho_saida, f"execucao_{config_id}.json")
        with open(caminho_json, 'w', encoding='utf-8') as arquivo_json:
            json.dump(info_execucao, arquivo_json, indent=2, ensure_ascii=False)
        print(f"Volumes de rótulos, centroides e metadados desta execução salvos em '{caminho_saida}' (prefixo '{config_id}').")