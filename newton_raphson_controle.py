import math
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# DBAR: Dados das barras do sistema elétrico (formato de lista de listas)
# Cada sublista representa uma barra com seus parâmetros:
# ----------------------------------------------------------------------
# [
#   número da barra (int),
#   tipo da barra (int):
#       0 = PQ (barra de carga),
#       1 = PV (barra geradora),
#       2 = barra swing (barra slack),
#   tensão nominal (pu) (float),
#   ângulo da tensão (graus ou radianos dependendo do contexto) (float),
#   potência ativa gerada Pg (pu) (float),
#   potência reativa gerada Qg (pu) (float),
#   potência reativa mínima Qmin (pu) (float),
#   potência reativa máxima Qmax (pu) (float),
#   barra controlada (int ou None): barra associada para controle de tensão (exemplo: para barras PV),
#   potência ativa da carga Pl (pu) (float),
#   potência reativa da carga Ql (pu) (float),
#   potência da injeção do gerador Sh (pu) (float), se aplicável,
#   área de controle ou área elétrica (int ou None),
#   fator de tensão Vf (float ou None): fator usado em alguns modelos de tensão
# ]

# ----------------------------------------------------------------------
# DLIN: Dados das linhas de transmissão (formato de lista de listas)
# Cada sublista representa uma linha com seus parâmetros:
# ----------------------------------------------------------------------
# [
#   barra "de" (int): barra inicial da linha,
#   barra "para" (int): barra final da linha,
#   circuito (int ou None): identificador de circuito da linha,
#   resistência R (ohms ou pu) (float),
#   reatância X (ohms ou pu) (float),
#   susceptância de linha Mvar (float),
#   tap (float): relação do transformador de tensão / tap changer,
#   Tmin (float ou None): valor mínimo permitido do tap,
#   Tmax (float ou None): valor máximo permitido do tap,
#   fase (float): ângulo de fase do transformador ou linha,
#   Bc (float ou None): susceptância de linha em shunt,
#   Cn (float ou None): parâmetro adicional (exemplo, capacidade nominal),
#   Ce (float ou None): parâmetro adicional (exemplo, capacidade emergencial),
#   Ns (int ou None): número de subconjuntos ou circuitos paralelos
# ]

# ----------------------------------------------------------------------
# Pbase: Potência base do sistema (em MW ou pu, conforme contexto)
# Usada para normalizar as potências no sistema, facilitando cálculos.

# ----------------------------------------------------------------------
# tol: Tolerância para critério de convergência do método (resíduo máximo
# aceito para potência ativa e reativa, por exemplo em MW ou pu)
# # Quando os resíduos ficarem abaixo dessa tolerância, o método para.
# tol = 0.003  # Critério de convergência para delta P e delta Q

# DBAR = [
#     [1, 2, 1.060, 0.0,   232.4, -16.9, -999, 999, None,   0.0,   0.0,   0.0, 1, None],
#     [2, 1, 1.045, -4.9,   40.0,  42.4, -10.0, 40.0, None, 21.7,  12.7,  0.0, 1, None],
#     [3, 0, 1.010, -12.0,   0.0,  0.0, -999, 999, None, 94.2,  22.0,  0.0, 1, None],
#     [4, 0, 1.019, -10.0,   0.0,   0.0, -999, 999, None,  47.8,  -3.9,  0.0, 1, None],
#     [5, 0, 1.020, -8.7,    0.0,   0.0, -999, 999, None,   15.6,   5.6,  0.0, 1, None],
#     [6, 1, 0.920, -2.7,    0.0,   0.0, -10.0, 10.0, None,   15.6,   5.6,  0.0, 1, None],
# ]

# # ----------------------------------------------------------------------
# # DLIN: Dados das linhas de transmissão
# # Cada entrada: [de, para, circuito, R (%), X (%), B (MVAr), tap, Tmin, Tmax, fase, Bc, Cn, Ce, Ns]
# # ----------------------------------------------------------------------
# DLIN = [
#     [1, 2, 1, 1.938, 5.917, 5.28, 1.0, None, None, 0.0, None, None, None, None],
#     [1, 5, 1, 5.403, 22.304, 4.92, 1.0, None, None, 0.0, None, None, None, None],
#     [1, 3, 1, 4.699, 19.797, 4.38, 1.0, None, None, 0.0, None, None, None, None],
#     [2, 4, 1, 5.811, 17.632, 3.40, 1.0, None, None, 0.0, None, None, None, None],
#     [2, 5, 1, 5.695, 17.388, 3.46, 1.0, None, None, 0.0, None, None, None, None],
#     [3, 6, 1, 2.395, 7.664, 3.10, 1.0, None, None, 0.0, None, None, None, None],
#     [5, 4, 1, 2.195, 7.464, 3.15, 1.5, None, None, 0.0, None, None, None, None],
# ]

# --------------------------------------------------------------------------------------

# DBAR = [
#     [1, 2, 1.060, 0.0,   100.0, -30.0, -999, 999, None,   0.0,   0.0,   0.0, 1, None],
#     [2, 1, 1.045, -5,   40.0,  10.0, -10.0, 12.0, None, 22.0,  12.0,  0.0, 1, None],
#     [3, 0, 1.010, -12.0,   0.0,  0.0, -999, 999, None, 95.0,  22.0,  0.0, 1, None],
# ]
# DLIN = [
#     [1, 2, 1, 2.0, 6.0, 6.0, 1.0, None, None, 0.0, None, None, None, None],
#     [1, 3, 1, 4.5, 20.0, 5.0, 1.0, None, None, 0.0, None, None, None, None],
#     [2, 3, 1, 5.0, 25.0, 4.0, 1.0, None, None, 0.0, None, None, None, None],
# ]
# ------------------------------------------------------------------------------

# IEEE 14 Bus Test Case - Dados formatados em Python
# Conversão baseada nos dados fornecidos no formato PWF, com valores percentuais já multiplicados por 100.

# ----------------------------------------------------------------------
# DBAR: Dados das barras do sistema elétrico
# Cada entrada: [número, tipo, V (pu), ângulo (°), Pg (MW), Qg (MVAr), Qmin, Qmax, barra controlada,
#                Pl (MW), Ql (MVAr), Sh (MW), área, fator de tensão Vf]
# ----------------------------------------------------------------------
DBAR = [
    [1, 2, 1.0, 0.0,   232.4, -16.9, -999, 999, None,   0.0,   0.0,   0.0, 1, None],
    [2, 1, 1.0, 0.0,  40.0,  42.4, -40.0, 50.0, None, 21.7,  12.7,  0.0, 1, None],
    [3, 1, 1.0, 0.0,   0.0,  23.4,   0.0, 40.0, None, 94.2,  19.0,  0.0, 1, None],
    [4, 0, 1.0, 0.0,   0.0,   0.0, -999, 999, None,  47.8,  -3.9,  0.0, 1, None],
    [5, 0, 1.0, 0.0,    0.0,   0.0, -999, 999, None,   7.6,   1.6,  0.0, 1, None],
    [6, 1, 1.0, 0.0,   0.0,  12.2,  -6.0, 24.0, None, 11.2,   7.5,  0.0, 1, None],
    [7, 0, 1.0, 0.0,   0.0,   0.0, -999, 999, None,   0.0,   0.0,  0.0, 1, None],
    [8, 1, 1.0, 0.0,   0.0,  17.4,  -6.0, 24.0, None,  0.0,   0.0,  0.0, 1, None],
    [9, 0, 1.0, 0.0,   0.0,   0.0, -999, 999, None,  29.5,  16.6, 19.0, 1, None],
    [10, 0, 1.0, 0.0,  0.0,   0.0, -999, 999, None,   9.0,   5.8,  0.0, 1, None],
    [11, 0, 1.0, 0.0, 0.0,   0.0, -999, 999, None,   3.5,   1.8,  0.0, 1, None],
    [12, 0, 1.0, 0.0,  0.0,   0.0, -999, 999, None,   6.1,   1.6,  0.0, 1, None],
    [13, 0, 1.0, 0.0,  0.0,   0.0, -999, 999, None,  13.5,   5.8,  0.0, 1, None],
    [14, 0, 1.0, 0.0,  0.0,   0.0, -999, 999, None,  14.9,   5.0,  0.0, 1, None]
]

# ----------------------------------------------------------------------
# DLIN: Dados das linhas de transmissão
# Cada entrada: [de, para, circuito, R (%), X (%), B (MVAr), tap, Tmin, Tmax, fase, Bc, Cn, Ce, Ns]
# ----------------------------------------------------------------------
DLIN = [
    [1, 2, 1, 1.938, 5.917, 5.28, 1.0, None, None, 0.0, None, None, None, None],
    [1, 5, 1, 5.403, 22.304, 4.92, 1.0, None, None, 0.0, None, None, None, None],
    [2, 3, 1, 4.699, 19.797, 4.38, 1.0, None, None, 0.0, None, None, None, None],
    [2, 4, 1, 5.811, 17.632, 3.40, 1.0, None, None, 0.0, None, None, None, None],
    [2, 5, 1, 5.695, 17.388, 3.46, 1.0, None, None, 0.0, None, None, None, None],
    [3, 4, 1, 6.701, 17.103, 1.28, 1.0, None, None, 0.0, None, None, None, None],
    [4, 5, 1, 1.335, 4.211, 0.0, 1.0, None, None, 0.0, None, None, None, None],
    [4, 7, 1, 0.0, 20.912, 0.0, 0.978, None, None, 0.0, None, None, None, None],
    [4, 9, 1, 0.0, 55.618, 0.0, 0.969, None, None, 0.0, None, None, None, None],
    [5, 6, 1, 0.0, 25.202, 0.0, 0.932, None, None, 0.0, None, None, None, None],
    [6, 11, 1, 9.498, 19.890, 0.0, 1.0, None, None, 0.0, None, None, None, None],
    [6, 12, 1, 12.291, 25.581, 0.0, 1.0, None, None, 0.0, None, None, None, None],
    [6, 13, 1, 6.615, 13.027, 0.0, 1.0, None, None, 0.0, None, None, None, None],
    [7, 8, 1, 0.0, 17.615, 0.0, 1.0, None, None, 0.0, None, None, None, None],
    [7, 9, 1, 0.0, 11.001, 0.0, 1.0, None, None, 0.0, None, None, None, None],
    [9, 10, 1, 3.181, 8.450, 0.0, 1.0, None, None, 0.0, None, None, None, None],
    [9, 14, 1, 12.711, 27.038, 0.0, 1.0, None, None, 0.0, None, None, None, None],
    [10, 11, 1, 8.205, 19.207, 0.0, 1.0, None, None, 0.0, None, None, None, None],
    [12, 13, 1, 22.092, 19.988, 0.0, 1.0, None, None, 0.0, None, None, None, None],
    [13, 14, 1, 17.093, 34.802, 0.0, 1.0, None, None, 0.0, None, None, None, None]
]

# ----------------------------------------------------------------------
# Potência base do sistema (MW)
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# Tolerância para critério de convergência
# ----------------------------------------------------------------------
tol = 0.01  # em pu

# Autor: Gonçalo Fontenele
# Data de criação: 18/05/2025
# Função: montar_ybarra
# Descrição: monta a matriz de admitância nodal Ybarra (completa), incluindo linhas com transformadores,
# defasagens angulares, admitâncias série e shunt por linha e por barra. Retorna também as partes G e B.

Pbase = 100  # Potência base típica para o sistema IEEE 14 barras
controle_PV = False #True = ativado; Falso=desativado.

def montar_ybarra(NBAR, NLIN, DE, PARA, R, X, BSH, TAP, DEFAS, SHUNT):
    """
    Monta a matriz de admitância Ybarra do sistema.

    Parâmetros:
    - NBAR: número de barras
    - NLIN: número de linhas
    - DE, PARA: listas de barras de origem e destino de cada linha (indexadas a partir de 1)
    - R, X: listas de resistências e reatâncias (pu)
    - BSH: susceptância shunt (pu)
    - TAP: relação de transformação dos transformadores (se for linha normal, TAP = 1)
    - DEFAS: defasagem angular dos trafos (rad)
    - SHUNT: admitância shunt em cada barra (pu)

    Retorna:
    - Ybarra: matriz de admitância completa
    - G: parte real da matriz
    - B: parte imaginária da matriz
    """

    # Inicialização da matriz Ybarra como matriz NBAR x NBAR de números complexos
    Ybarra = np.zeros((NBAR, NBAR), dtype=complex)

    # Cálculo da admitância série de cada linha (Ykm = 1 / Z)
    Ykm = 1 / (R + 1j * X)

    # Cálculo da admitância shunt (representada por susceptância Bsh) de cada linha
    Bsh = 1j * BSH

    # Loop para montagem da matriz Ybarra (somente componentes das linhas)
    for k in range(NLIN):
        x = int(DE[k]) - 1      # índice da barra DE (ajuste de indexação para Python)
        y = int(PARA[k]) - 1    # índice da barra PARA
        tap_k = TAP[k]          # relação de transformação (TAP)
        defas_k = DEFAS[k]      # defasagem angular do trafo (rad)

        # Adição do termo série e shunt da linha na diagonal da barra "de"
        Ybarra[x, x] += Bsh[k] + (1 / tap_k**2) * Ykm[k]

        # Adição do termo série e shunt da linha na diagonal da barra "para"
        Ybarra[y, y] += Bsh[k] + Ykm[k]

        # Termos fora da diagonal com ajuste de tap e defasagem (ângulo)
        Ybarra[x, y] -= Ykm[k] * (1 / tap_k) * np.exp(-1j * defas_k)
        Ybarra[y, x] -= Ykm[k] * (1 / tap_k) * np.exp(1j * defas_k)

    # Adição das admitâncias shunt associadas a cada barra diretamente (elementos da diagonal)
    for k in range(NBAR):
        Ybarra[k, k] += 1j * SHUNT[k]

    # Extração das partes real (G) e imaginária (B) da matriz Ybarra
    G = np.real(Ybarra)
    B = np.imag(Ybarra)

    return Ybarra, G, B

# Autor: Gonçalo Fontenele
# Data de criação: 18/05/2025
# Função: montar_matriz_jacobiana
# Descrição: monta a matriz Jacobiana do método de Newton-Raphson para fluxo de potência,
# considerando barras tipo PQ, PV e Swing. A Jacobiana é composta pelas submatrizes H, N, M, L.

def montar_matriz_jacobiana(NBAR, V, TETA, Pcalc, Qcalc, G, B, TIPO):
    # Inicialização das submatrizes H, N, M, L com zeros (todas são NBAR x NBAR)
    H = np.zeros((NBAR, NBAR))  # Derivadas de P em relação a ângulos (∂P/∂θ)
    N = np.zeros((NBAR, NBAR))  # Derivadas de P em relação a tensões (∂P/∂V)
    M = np.zeros((NBAR, NBAR))  # Derivadas de Q em relação a ângulos (∂Q/∂θ)
    L = np.zeros((NBAR, NBAR))  # Derivadas de Q em relação a tensões (∂Q/∂V)

    # Loop para montar as derivadas de todas as submatrizes
    for k1 in range(NBAR):
        for k2 in range(NBAR):
            if k1 == k2:
                # Elementos diagonais (derivadas parciais considerando a própria barra)
                H[k1, k1] = -(Qcalc[k1] + V[k1]**2 * B[k1, k1])
                N[k1, k1] = (1 / V[k1]) * (Pcalc[k1] + V[k1]**2 * G[k1, k1])
                M[k1, k1] = Pcalc[k1] - V[k1]**2 * G[k1, k1]
                L[k1, k1] = (1 / V[k1]) * (Qcalc[k1] - V[k1]**2 * B[k1, k1])

                # Ajustes com base no tipo da barra
                if TIPO[k1] == 2:  # Barra swing: ângulo e tensão fixos
                    H[k1, k1] = 1e10   # Impede alterações em ∆θ
                    L[k1, k1] = 1e10   # Impede alterações em ∆V
                if TIPO[k1] == 1:  # Barra PV: ângulo e potência ativa fixos
                    L[k1, k1] = 1e10   # Impede alterações em ∆V (Q especificado)
            else:
                # Elementos fora da diagonal (entre barras diferentes)
                delta_theta = TETA[k1] - TETA[k2]

                # Derivadas parciais conforme a equação do fluxo de potência
                H[k1, k2] = V[k1] * V[k2] * (G[k1, k2] * np.sin(delta_theta) - B[k1, k2] * np.cos(delta_theta))
                N[k1, k2] = V[k1] * (G[k1, k2] * np.cos(delta_theta) + B[k1, k2] * np.sin(delta_theta))
                M[k1, k2] = -V[k1] * V[k2] * (G[k1, k2] * np.cos(delta_theta) + B[k1, k2] * np.sin(delta_theta))
                L[k1, k2] = V[k1] * (G[k1, k2] * np.sin(delta_theta) - B[k1, k2] * np.cos(delta_theta))

    # Montagem da matriz Jacobiana 2*NBAR x 2*NBAR com as 4 submatrizes
    Jac = np.block([[H, N],   # Parte superior (∂P/∂θ | ∂P/∂V)
                    [M, L]])  # Parte inferior (∂Q/∂θ | ∂Q/∂V)

    return Jac

def verificar_limites_Q(PV, Qcalc, QM, QN, QG, TIPO, Pbase, iter_bloqueio_reversao, i):
    for idx in PV:
        if Qcalc[idx] > QM[idx]:
            print(f'Iteração {i}: BARRA {idx+1} virou PQ (Qcalc = {Qcalc[idx]*Pbase:.4f} > Qmax = {QM[idx]*Pbase:.4f})')
            QG[idx] = QM[idx]
            TIPO[idx] = 0
            iter_bloqueio_reversao[idx] = i
        elif Qcalc[idx] < QN[idx]:
            print(f'Iteração {i}: BARRA {idx+1} virou PQ (Qcalc = {Qcalc[idx]*Pbase:.4f} < Qmin = {QN[idx]*Pbase:.4f})')
            QG[idx] = QN[idx]
            TIPO[idx] = 0
            iter_bloqueio_reversao[idx] = i
    return QG, TIPO, iter_bloqueio_reversao

def atualizar_listas_tipo(TIPO):
    PV = np.where(TIPO == 1)[0]
    PQ = np.where(TIPO == 0)[0]
    return PV, PQ, len(PV), len(PQ)

def reverter_tipo_barras(TIPO_ORIGINAL, TIPO, Qcalc, QM, QN, iter_bloqueio_reversao, i, V, VREF, QG):
    for idx in range(len(TIPO)):
        if TIPO_ORIGINAL[idx] == 1 and TIPO[idx] == 0:
            if i > iter_bloqueio_reversao[idx]:
                if Qcalc[idx] < QN[idx] or Qcalc[idx] > QM[idx]:
                    continue
                msg = f'Iteração {i}: BARRA {idx+1} voltou a ser PV (Q dentro dos limites após iteração de bloqueio): Q = {Qcalc[idx]*Pbase:.4f}'
                print(msg)
                TIPO[idx] = 1
                V[idx] = VREF[idx]  # <<< restaura o valor original
                QG[idx] = Qcalc[idx]     # Atualiza QG corretamente
    return TIPO, QG

def controle_tensao_reversivel(i, controle_PV, PV, Qcalc, QM, QN, QG, TIPO, TIPO_ORIGINAL, Pbase, iter_bloqueio_reversao):
    if i > 0 and controle_PV:
        QG, TIPO, iter_bloqueio_reversao = verificar_limites_Q(
            PV, Qcalc, QM, QN, QG, TIPO, Pbase, iter_bloqueio_reversao, i)
        PV, PQ, NPV, NPQ = atualizar_listas_tipo(TIPO)
        return QG, TIPO, PV, PQ, NPV, NPQ, iter_bloqueio_reversao
    else:
        return QG, TIPO, PV, np.where(TIPO == 0)[0], len(PV), len(np.where(TIPO == 0)[0]), iter_bloqueio_reversao

def newton_raphson_flow(DBAR, DLIN, Pbase = 1.0, tolerancia = 0.003, iteracao_max = 20):
    #Gerar relatório para permitir avaliar os cálculos
    relatorio_iteracoes = []
    VREF = np.array([row[2] for row in DBAR])  # Vetor auxiliar par o controle de tensão

    # Número de barras (NBAR)
    NBAR = len(DBAR)  # número de linhas da lista DBAR

    # Número de linhas (NLIN) e número de colunas (AUX) da matriz DLIN
    NLIN = len(DLIN)        # número de linhas da lista DLIN

    # --- Separação de DBAR em vetores numpy (ordenados por índice) ---
    # DBAR contém os dados das barras do sistema

    TIPO = np.array([row[1] for row in DBAR])            # Tipo da barra (1 = PQ, 2 = PV, 3 = Slack)
    V = np.array([row[2] for row in DBAR])               # Módulo da tensão (em pu)
    TETA = np.array([row[3] * math.pi / 180 for row in DBAR])  # Ângulo da tensão (graus → rad)
    PG = np.array([row[4] / Pbase for row in DBAR])      # Potência ativa gerada (MW → pu)
    QG = np.array([row[5] / Pbase for row in DBAR])      # Potência reativa gerada (MVAr → pu)
    QN = np.array([row[6] / Pbase for row in DBAR])      # Limite inferior da geração reativa (MVAr → pu)
    QM = np.array([row[7] / Pbase for row in DBAR])      # Limite superior da geração reativa (MVAr → pu)
    PD = np.array([row[9] / Pbase for row in DBAR])      # Potência ativa demandada (MW → pu)
    QD = np.array([row[10] / Pbase for row in DBAR])     # Potência reativa demandada (MVAr → pu)
    SHUNT = np.array([row[11] / Pbase for row in DBAR])  # Susceptância do shunt (MVAr → pu)

    # --- Separação de DLIN em vetores numpy (ordenados por índice) ---
    # DLIN contém os dados das linhas do sistema

    DE = np.array([row[0] for row in DLIN])              # Barra de origem da linha
    PARA = np.array([row[1] for row in DLIN])            # Barra de destino da linha
    R = np.array([row[3] / 100 for row in DLIN])         # Resistência série da linha (% → pu)
    X = np.array([row[4] / 100 for row in DLIN])         # Reatância série da linha (% → pu)
    BSH = np.array([(row[5] / 2) / Pbase for row in DLIN])  # Susceptância total da linha (dividida entre as extremidades e normalizada)
    TAP = np.array([row[6] for row in DLIN])             # Tap da linha (se houver transformador, normalmente ≠ 1)
    DEFAS = np.array([row[9] for row in DLIN])           # Defasagem angular associada ao tap (em graus ou rad, conforme o caso)

    # Seleção das Barras PV (TIPO == 1)
    PV = np.where(TIPO == 1)[0]
    NPV = len(PV)

    # Seleção das Barras PQ (TIPO == 0)
    PQ = np.where(TIPO == 0)[0]
    NPQ = len(PQ)

    # Matriz de Fluxo nas Linhas (5 colunas)
    FLUXO = np.zeros((NLIN, 5))
    FLUXO[:, 2] = np.array([linha[12] if len(linha) > 14 else 0 for linha in DLIN])  # PkmMAX

    # Inicialização
    i = 0  # Número de Iterações

    convergiu = False  # Flag para verificar divergência
    TIPO_ORIGINAL = TIPO.copy()
    iter_bloqueio_reversao = np.full(NBAR, -1)  # -1 = sem bloqueio

    while not convergiu and i < iteracao_max:
        # Tensões em coordenadas retangulares (TETA em radianos)
        x = V * np.cos(TETA)
        y = V * np.sin(TETA)
        Vret = x + 1j * y  # Forma retangular da tensão

        # Valores Especificados de Potência (Líquido)
        Pesp = PG - PD
        Qesp = QG - QD

        # Montagem da matriz Ybarra
        Ybarra, G, B = montar_ybarra(NBAR, NLIN, DE, PARA, R, X, BSH, TAP, DEFAS, SHUNT)

        # NOTA:
        # AQUI SERÁ NECESSÁRIO ALTERAR A YBARRA SEMPRE QUE TIVER UM CONTROLE ACIONADO
        # ACREDITO QUE DÊ PARA RODAR ALGUMAS ITERAÇÕES E TER DUAS TOLERÂNCIAS
        # CHEGANDO EM 30 % DA TOLERÂNCIA MÁXIMA, SOLTARIA OS CONTROLES

        # Correntes injetadas: I = Ybarra * V
        I = Ybarra @ Vret  # Produto matricial

        # Potência complexa injetada: S = V * conj(I)
        S = Vret * np.conj(I)
        Pcalc = np.real(S)  # Potência ativa calculada
        Qcalc = np.imag(S)  # Potência reativa calculada

        # Controle de Q nas barras PV após a 1ª iteração
        QG, TIPO, PV, PQ, NPV, NPQ, iter_bloqueio_reversao = controle_tensao_reversivel(i, controle_PV, PV, Qcalc, QM, QN, QG, TIPO, TIPO_ORIGINAL, Pbase, iter_bloqueio_reversao)
        TIPO, QG = reverter_tipo_barras(TIPO_ORIGINAL, TIPO, Qcalc, QM, QN, iter_bloqueio_reversao, i, V, VREF, QG)

                # Inicialização dos resíduos
        delta_P = Pesp - Pcalc
        delta_Q = Qesp - Qcalc

        # Ajuste dos resíduos com base no tipo de barra
        for k in range(NBAR):
            if TIPO[k] == 2:  # Slack (tipo 2): não há resíduos
                delta_P[k] = 0
                delta_Q[k] = 0
            elif TIPO[k] == 1:  # PV (tipo 1): não há resíduo de Q
                delta_Q[k] = 0

        # Vetor de resíduos (coluna)
        delta_Y = np.concatenate([delta_P, delta_Q]).reshape(-1, 1)

        # Erro máximo (critério de convergência)
        MAX_Y = np.max(np.abs(delta_Y))

        if MAX_Y < tolerancia:
            convergiu = True
            break

        i += 1  # Incrementa o contador de iterações

        # Matriz Jacobiana
        Jac = montar_matriz_jacobiana(NBAR, V, TETA, Pcalc, Qcalc, G, B, TIPO)

        # NOTA:
        # PENSO EM TALVEZ ADICIONAR INFORMAÇÕES APÓS MONTAR A JACOBIANA PADRÃO
        # ELA SEMPRE TERÁ SEU PADRÃO IGUAL, SÓ ALTERADO PELA Y BARRA QUE JÁ FOI ALTERADA ANTES
        # DESSA FORMA, PODEMOS SEMPRE ADICIONAR OS CONTROLES EM SEGUIDA
        # TALVEZ COM UMA FUNÇÃO adicionar_controles_jacobiana

        # delta_Y deve ser um vetor coluna numpy com dimensão (2*NBAR, 1)
        # Jac é a matriz Jacobiana 2*NBAR x 2*NBAR

        # Resolução do sistema linear: Jac * delta_SOLUCAO = delta_Y
        delta_SOLUCAO = np.linalg.solve(Jac, delta_Y).flatten()

        # Atualização dos vetores TETA e V
        TETA += delta_SOLUCAO[0:NBAR]            # primeiros NBAR elementos são delta_TETA
        V += delta_SOLUCAO[NBAR:2*NBAR]             # próximos NBAR elementos são delta_V
        relatorio_iteracoes.append({
            'iteracao': i,
            'Pesp': Pesp.copy(),
            'Qesp': Qesp.copy(),
            'Pcalc': Pcalc.copy(),
            'Qcalc': Qcalc.copy(),
            'delta_P': delta_P.copy(),
            'delta_Q': delta_Q.copy(),
            'delta_TETA': delta_SOLUCAO[0:NBAR].copy(),
            'delta_V': delta_SOLUCAO[NBAR:2*NBAR].copy(),
            'TETA': TETA.copy(),
            'V': V.copy()
            })
    
    linhas_relatorio = []
    for item in relatorio_iteracoes:
        iter_num = item['iteracao']
        for k in range(NBAR):
            linhas_relatorio.append({
                'Iteracao': iter_num,
                'Barra': k + 1,
                'Pesp': item['Pesp'][k],
                'Qesp': item['Qesp'][k],
                'Pcalc': item['Pcalc'][k],
                'Qcalc': item['Qcalc'][k],
                'delta_P': item['delta_P'][k],
                'delta_Q': item['delta_Q'][k],
                'delta_TETA (rad)': item['delta_TETA'][k],
                'delta_V': item['delta_V'][k],
                'TETA (graus)': item['TETA'][k] * 180 / np.pi,
                'V': item['V'][k]
            })

    df_relatorio = pd.DataFrame(linhas_relatorio)
    df_relatorio.to_excel('relatorio_fluxo_potencia.xlsx', index=False)
    print('Relatório de iterações salvo em: relatorio_fluxo_potencia.xlsx')

    for k in range(NLIN):
        K = DE[k]-1
        M = PARA[k]-1

        fluxo_1 = -(
            (TAP[k] * V[K])**2 * G[K, M]
            - (TAP[k] * V[K]) * V[M] * G[K, M] * np.cos(TETA[K] - TETA[M])
            - (TAP[k] * V[K]) * V[M] * B[K, M] * np.sin(TETA[K] - TETA[M])
        ) * Pbase

        fluxo_2 = -(
            (TAP[k] * V[M])**2 * G[M, K]
            - (TAP[k] * V[M]) * V[K] * G[M, K] * np.cos(TETA[M] - TETA[K])
            - (TAP[k] * V[M]) * V[K] * B[M, K] * np.sin(TETA[M] - TETA[K])
        ) * Pbase

        FLUXO[k, 0] = fluxo_1  # coluna 1 (Python index 0)
        FLUXO[k, 1] = fluxo_2  # coluna 2 (Python index 1)

        if fluxo_1 > fluxo_2:
            FLUXO[k, 3] = FLUXO[k, 2] - abs(fluxo_1)  # coluna 4 = coluna 3 - |fluxo_1|
            FLUXO[k, 4] = abs(fluxo_1)                 # coluna 5 = |fluxo_1|
        else:
            FLUXO[k, 3] = FLUXO[k, 2] - abs(fluxo_2)  # coluna 4 = coluna 3 - |fluxo_2|
            FLUXO[k, 4] = abs(fluxo_2)                 # coluna 5 = |fluxo_2|

 
    if not convergiu:
        print('O caso Divergiu')
    else:
        print('='*108)
        print(f'- Número de Iterações: {i}')
        print('='*108)
        print(f'- Resíduo Máximo: {MAX_Y:.6g} < Tolerância de {tolerancia:.3f}')
        print('='*108)
        print('- Dados Finais de Barra (pu):\n')
        print(f'{"Nº":>2} {"Tipo":>4} {"V":>6} {"Ang(°)":>8} {"PG":>8} {"QG":>8} {"Qmín":>8} {"Qmáx":>8} {"Pd":>8} {"Qd":>8}')
       
        # Monta matriz de dados para impressão
        dados_barra = np.column_stack((
            np.arange(1, NBAR+1),              # Nº
            TIPO,                             # Tipo
            V,                                # Tensão
            TETA * 180 / np.pi,               # Ângulo em graus
            (Pcalc + PD) * Pbase,             # PG
            (Qcalc + QD) * Pbase,             # QG
            QN * Pbase,                      # Qmín
            QM * Pbase,                      # Qmáx
            PD * Pbase,                      # Pdemanda
            QD * Pbase                       # Qdemanda
        ))

        # Imprime linha a linha formatada
        for linha in dados_barra:
            print(f'{int(linha[0]):<4} {int(linha[1]):<6} {linha[2]:<8.4f} {linha[3]:<8.2f}{linha[4]:<8.4f} {linha[5]:<8.4f} {linha[6]:<8.4f} {linha[7]:<8.4f}  {linha[8]:<8.4f} {linha[9]:<8.4f}')

        print('='*108)
        print('Fluxo de Potência Ativa entre as Barras')
        print('='*108)
        print(f'\t{"DE":<6}\t{"PARA":<6}\t{"DE-PARA":<10}\t{"PARA-DE":<10}\t{"FLUXO MAX":<10}\t{"FOLGA":<10}')

        # Imprime dados de fluxo: DE, PARA, FLUXO[:,1], FLUXO[:,2], FLUXO[:,3], FLUXO[:,4]
        for i in range(len(DE)):
            print(f'\t{int(DE[i]):<6}\t{int(PARA[i]):<6}\t{FLUXO[i,0]:<10.4f}\t{FLUXO[i,1]:<10.4f}\t{FLUXO[i,2]:<10.4f}\t{FLUXO[i,3]:<10.4f}')

        print('='*108)

    return V, TETA, FLUXO

tensoes, thetas, fluxos = newton_raphson_flow(DBAR, DLIN, Pbase = Pbase, tolerancia = tol)