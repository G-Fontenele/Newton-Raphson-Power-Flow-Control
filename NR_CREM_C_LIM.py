import math
import numpy as np
import pandas as pd
import copy

Controle_REM = False #ativa/desativa controle remoto de tensão - Utilizado em Newton_raphson após declaração de QBAR
Controle_PV = False #True = ativado; Falso=desativado (Exceto barras de controle remoto de tensão).

Pbase = 100  # Potência base típica para o sistema IEEE 14 barras
tol = 0.01/Pbase  # em pu
tol_Vcontrolada = 0.6 #tolerância diferenciada para deltaV das barras controladas por CREM.

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
# DBAR: Dados das barras do sistema elétrico
# Cada entrada: [número, tipo, V (pu), ângulo (°), Pg (MW), Qg (MVAr), Qmin, Qmax, barra controlada,
#                Pl (MW), Ql (MVAr), Sh (MW), área, fator de tensão Vf]
# ----------------------------------------------------------------------
DBAR = [
    [1, 2, 1.060, 0.0,   232.4, -16.9, -999, 999, None,   0.0,   0.0,   0.0, 1, None],
    [2, 1, 1.045, 0.0,   40.0,  42.4, -40.0, 50.0, None, 21.7,  12.7,  0.0, 1, None],
    [3, 1, 1.010, 0.0,   0.0,  23.4,   0.0, 40.0, None, 94.2,  19.0,  0.0, 1, None],
    [4, 0, 1.019, 0.0,   0.0,   0.0, -999, 999, None,  47.8,  -3.9,  0.0, 1, None],
    [5, 0, 1.020, 0.0,    0.0,   0.0, -999, 999, None,   7.6,   1.6,  0.0, 1, None],
    [6, 1, 1.070, 0.0,   0.0,  12.2,  -6, 24, 12, 11.2,   7.5,  0.0, 1, None],
    [7, 0, 1.000, 0.0,   0.0,   0.0, -999, 999, None,   0.0,   0.0,  0.0, 1, None],
    [8, 1, 1.090, 0.0,   0.0,  17.4,  -6, 24, 7,  0.0,   0.0,  0.0, 1, None],
    [9, 0, 1.056, 0.0,   0.0,   0.0, -999, 999, None,  29.5,  16.6, 19.0, 1, None],
    [10, 0, 1.051, 0.0,  0.0,   0.0, -999, 999, None,   9.0,   5.8,  0.0, 1, None],
    [11, 0, 1.057, 0.0,  0.0,   0.0, -999, 999, None,   3.5,   1.8,  0.0, 1, None],
    [12, 0, 1.070, 0.0,  0.0,   0.0, -999, 999, None,   6.1,   1.6,  0.0, 1, None],
    [13, 0, 1.050, 0.0,  0.0,   0.0, -999, 999, None,  13.5,   5.8,  0.0, 1, None],
    [14, 0, 1.036, 0.0,  0.0,   0.0, -999, 999, None,  14.9,   5.0,  0.0, 1, None]
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

def montar_ybarra(NBAR, NLIN, DE, PARA, R, X, BSH, TAP, DEFAS, SHUNT):
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

def montar_matriz_jacobiana(NBAR, V, TETA, Pcalc, Qcalc, G, B, TIPO, B_controle, B_controlada, N_controlada, Cv, Ch):
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

    if Controle_REM:
        for n in range(N_controlada):
            Cv[NBAR + B_controle[n]-1 , n] = -1
            Ch[n, NBAR + B_controlada[n]-1] = -1

        # Montagem da matriz Jacobiana 2*NBAR x 2*NBAR com as 4 submatrizes
        Jac_superior = np.hstack([np.vstack([H, M]), np.vstack([N, L]), Cv])
        Jac = np.vstack([Jac_superior, Ch])
    else:
        Jac = np.hstack([np.vstack([H, M]), np.vstack([N, L])])

    return Jac

def verificar_limites_iniciais(DBAR):
    """    Verifica e corrige valores iniciais de QG que violam Qmin/Qmax
    Retorna DBAR corrigido e lista de alterações realizadas"""
    alteracoes = []
    for barra in DBAR:
        num_barra = barra[0]
        tipo = barra[1]
        qg = barra[5]
        qmin = barra[6]
        qmax = barra[7]
        
        if tipo == 1:  # Apenas para barras PV
            if qg > qmax:
                alteracoes.append(f"Barra {num_barra}: QG inicial {qg} > Qmax {qmax} - Qg inicial ajustado para Qmax")
                barra[5] = qmax
            elif qg < qmin:
                alteracoes.append(f"Barra {num_barra}: QG inicial {qg} < Qmin {qmin} - Qg inicial ajustado para Qmin")
                barra[5] = qmin
    
    return DBAR, alteracoes

def verificar_limites_Q(PV, Qcalc, QM, QN, QG, TIPO, Pbase, iter_bloqueio_reversao, i, violou_qmax, violou_qmin, QD):
    for idx in PV:
        if TIPO[idx] != 1:  # Blindagem: só PV local
            continue

        if Qcalc[idx] + QD[idx] > QM[idx]:
            print(f'Iteração {i}: BARRA {idx+1} virou PQ (Qcalc = {(Qcalc[idx] + QD[idx])*Pbase:.4f} > Qmax = {QM[idx]*Pbase:.4f})')
            QG[idx] = QM[idx]
            TIPO[idx] = 0
            iter_bloqueio_reversao[idx] = i
            violou_qmax[idx] = True
            violou_qmin[idx] = False  
        elif Qcalc[idx] + QD[idx] < QN[idx]:
            print(f'Iteração {i}: BARRA {idx+1} virou PQ (Qcalc = {(Qcalc[idx] + QD[idx])*Pbase:.4f} < Qmin = {QN[idx]*Pbase:.4f})')
            QG[idx] = QN[idx]
            TIPO[idx] = 0
            iter_bloqueio_reversao[idx] = i
            violou_qmax[idx] = False
            violou_qmin[idx] = True
    return QG, TIPO, iter_bloqueio_reversao, violou_qmax, violou_qmin

def atualizar_listas_tipo(TIPO):
    PV = np.where(TIPO == 1)[0]
    PQ = np.where(TIPO == 0)[0]
    return PV, PQ, len(PV), len(PQ)

def reverter_tipo_barras(TIPO_ORIGINAL, TIPO, V, VREF, Qcalc, QG, violou_qmax, violou_qmin, iter_bloqueio_reversao, i):
    for idx in range(len(TIPO)):
        if TIPO_ORIGINAL[idx] != 1:  # Blindagem: originalmente não era PV
            continue
        if TIPO[idx] != 0:  # Só tenta reverter PQ para PV
            continue

        if i > iter_bloqueio_reversao[idx]:
            delta_V = VREF[idx] - V[idx]

            if violou_qmax[idx] and delta_V < 0:
                msg = f'Iteração {i}: BARRA {idx+1} voltou a ser PV (ΔV = {delta_V:.5f} < 0 e violou Qmax)'
            elif violou_qmin[idx] and delta_V > 0:
                msg = f'Iteração {i}: BARRA {idx+1} voltou a ser PV (ΔV = {delta_V:.5f} > 0 e violou Qmin)'
            else:
                continue  # Não atende aos critérios de reversão

            print(msg)
            TIPO[idx] = 1
            V[idx] = VREF[idx]
            QG[idx] = Qcalc[idx]
    return TIPO, QG

def controle_tensao_reversivel(i, Controle_PV, PV, Qcalc, QM, QN, QG, TIPO, TIPO_ORIGINAL, Pbase, iter_bloqueio_reversao, violou_qmax, violou_qmin, QD):
    if i > 0 and Controle_PV:
        QG, TIPO, iter_bloqueio_reversao, violou_qmax, violou_qmin = verificar_limites_Q(PV, Qcalc, QM, QN, QG, TIPO, Pbase, iter_bloqueio_reversao, i, violou_qmax, violou_qmin, QD)
        PV, PQ, NPV, NPQ = atualizar_listas_tipo(TIPO)
        return QG, TIPO, PV, PQ, NPV, NPQ, iter_bloqueio_reversao, violou_qmax, violou_qmin
    else:
        return QG, TIPO, PV, np.where(TIPO == 0)[0], len(PV), len(np.where(TIPO == 0)[0]), iter_bloqueio_reversao, violou_qmax, violou_qmin

def relatorio_transformadores(DE, PARA, FLUXO, TAP, LC, nomes_barras=None):
    """
    Imprime um relatório no estilo ANAREDE para transformadores/linhas com tap.

    Parâmetros:
    - DE, PARA: listas de barras de origem e destino
    - FLUXO: matriz com colunas [Pkm, Pmk, Pkm_max, folga, fluxo_utilizado, Qkm]
    - TAP: vetor de taps atuais
    - LC: lista de barras controladas (None se não houver controle)
    - nomes_barras: lista com nomes das barras (opcional)
    """

    print("\nRELATÓRIO DE TRANSFORMADORES COM CONTROLE DE TAP")
    print("X-------X-----------------X-----X-----------------X-----X-------X-------X--------X")
    print(" DE     NOME_ORIGEM       NC    PARA   NOME_DESTINO     MW     Mvar    MVA     TAP  TIPO")
    print("X-------X-----------------X-----X-----------------X-----X-------X-------X--------X")

    for k in range(len(DE)):
        de = DE[k]
        para = PARA[k]
        nome_de = nomes_barras[de - 1] if nomes_barras else f"Barra-{de:02d}"
        nome_para = nomes_barras[para - 1] if nomes_barras else f"Barra-{para:02d}"

        pkm = FLUXO[k, 0]
        qkm = FLUXO[k, 5] if FLUXO.shape[1] > 5 else 0.0
        mva = (pkm**2 + qkm**2)**0.5
        tap = TAP[k]
        tipo = "*" if LC[k] is not None else "F"  # * para variável, F para fixo

        print(f" {de:<7} {nome_de:<17} 1     {para:<5} {nome_para:<17} {pkm:7.1f} {qkm:7.1f} {mva:7.1f}  {tap:6.3f}   {tipo:>2}")

    print("X-------X-----------------X-----X-----------------X-----X-------X-------X--------X")
    return


def newton_raphson_flow(DBAR, DLIN, Pbase = 1.0, tolerancia = 0.003, tol_Vcontrolada = 0.03, iteracao_max = 20):
    # Número de barras (NBAR)
    NBAR = len(DBAR)  # número de linhas da lista DBAR

    # Número de linhas (NLIN) e número de colunas (AUX) da matriz DLIN
    NLIN = len(DLIN)        # número de linhas da lista DLIN

    # DBAR contém os dados das barras do sistema

    TIPO = np.array([row[1] for row in DBAR])            # Tipo da barra (1 = PQ, 2 = PV, 3 = Slack)
    V = np.array([row[2] for row in DBAR])               # Módulo da tensão (em pu)
    V_ESP = copy.deepcopy(V)    
    TETA = np.array([row[3] * math.pi / 180 for row in DBAR])  # Ângulo da tensão (graus → rad)
    TETA_ESP = copy.deepcopy(TETA)
    PG = np.array([row[4] / Pbase for row in DBAR])      # Potência ativa gerada (MW → pu)
    QG = np.array([row[5] / Pbase for row in DBAR])      # Potência reativa gerada (MVAr → pu)
    QN = np.array([row[6] / Pbase for row in DBAR])      # Limite inferior da geração reativa (MVAr → pu)
    QM = np.array([row[7] / Pbase for row in DBAR])      # Limite superior da geração reativa (MVAr → pu)
    BC = np.array([row[8] for row in DBAR])             # Barra controlada (se houver)
    PD = np.array([row[9] / Pbase for row in DBAR])      # Potência ativa demandada (MW → pu)
    QD = np.array([row[10] / Pbase for row in DBAR])     # Potência reativa demandada (MVAr → pu)
    SHUNT = np.array([row[11] / Pbase for row in DBAR])  # Susceptância do shunt (MVAr → pu)

    
    if Controle_REM:
        B_controle = []
        B_controlada = []
        Vb_controlada = []

        for barra in DBAR:
            num_barra = barra[0]
            barra_controlada = barra[8]
            if barra_controlada is not None:
                B_controle.append(num_barra)
                B_controlada.append(barra_controlada)
                idx_controlada = next(i for i, b in enumerate(DBAR) if b[0] == barra_controlada)
                Vb_controlada.append(DBAR[idx_controlada][2])

        N_controle = len(B_controle)
        N_controlada = len(B_controlada)
        Cv = np.zeros((2*NBAR, N_controlada))
        Ch = np.zeros((N_controlada, 2*NBAR + N_controlada))

        # Agora sim: alteração dos tipos das barras conforme controle
        for b in range(N_controle):
            if TIPO[B_controle[b]-1] != 2:  # Evita sobrescrever a barra slack
                TIPO[B_controle[b]-1] = 3  # Tipo 3 = barra controladora remota

        for b in range(N_controlada):
            if TIPO[B_controlada[b]-1] != 2:
                TIPO[B_controlada[b]-1] = 4  # Tipo 4 = barra controlada remotamente
    else:
        B_controle = []
        B_controlada = []
        Vb_controlada = []
        N_controle = 0
        N_controlada = 0
        Cv = None
        Ch = None

    # Se o controle de limites de Q estiver desativado, zera os limites para evitar reversão
    if not Controle_PV:
        QN = np.full(NBAR, -999.0)
        QM = np.full(NBAR, 999.0)

    # DLIN contém os dados das linhas do sistema

    DE = np.array([row[0] for row in DLIN])              # Barra de origem da linha
    PARA = np.array([row[1] for row in DLIN])            # Barra de destino da linha
    R = np.array([row[3] / 100 for row in DLIN])         # Resistência série da linha (% → pu)
    X = np.array([row[4] / 100 for row in DLIN])         # Reatância série da linha (% → pu)
    BSH = np.array([(row[5] / 2) / Pbase for row in DLIN])  # Susceptância total da linha (dividida entre as extremidades e normalizada)
    TAP = np.array([row[6] for row in DLIN])             # Tap da linha (se houver transformador, normalmente ≠ 1)
    DEFAS = np.array([row[9] for row in DLIN])           # Defasagem angular associada ao tap (em graus ou rad, conforme o caso)
    LC = np.array([row[10] for row in DLIN])            # Barra/Linha controlada

    # Seleção das Barras PV (TIPO == 1)
    PV = np.where(TIPO == 1)[0]
    NPV = len(PV)

    # Seleção das Barras PQ (TIPO == 0)
    PQ = np.where(TIPO == 0)[0]
    NPQ = len(PQ)

    # Matriz de Fluxo nas Linhas (5 colunas)
    FLUXO = np.zeros((NLIN, 6))  # Adiciona uma coluna extra para o fluxo reativo
    FLUXO[:, 2] = np.array([linha[12] if len(linha) > 14 else 0 for linha in DLIN])  # PkmMAX

    DBAR, alteracoes_Qinicial = verificar_limites_iniciais(DBAR)
    for msg in alteracoes_Qinicial:
        print(msg)

    # Inicialização
    i = 0  # Número de Iterações

    convergiu = False  # Flag para verificar divergência

    TIPO_ORIGINAL = TIPO.copy()
    VREF = V.copy()

    iter_bloqueio_reversao = np.full(NBAR, -1)      # Vetor com a última iteração de reversão para cada barra
    violou_qmax = np.full(NBAR, False, dtype=bool)  # Flags para saber por qual limite foi revertida
    violou_qmin = np.full(NBAR, False, dtype=bool)
    #Relatório em excel para identificar onde está o erro de não convergir.
    relatorio_iteracoes = []

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

        # Correntes injetadas: I = Ybarra * V
        I = Ybarra @ Vret  # Produto matricial

        # Potência complexa injetada: S = V * conj(I)
        S = Vret * np.conj(I)
        Pcalc = np.real(S)  # Potência ativa calculada
        Qcalc = np.imag(S)  # Potência reativa calculada

        # CONTROLE BARRAS PV:Controle de Q nas barras PV após a 1ª iteração.
        if Controle_PV:
            QG, TIPO, PV, PQ, NPV, NPQ, iter_bloqueio_reversao, violou_qmax, violou_qmin = controle_tensao_reversivel(i, Controle_PV, PV, Qcalc, QM, QN, QG, TIPO, TIPO_ORIGINAL, Pbase, iter_bloqueio_reversao, violou_qmax, violou_qmin, QD)
            TIPO, QG = reverter_tipo_barras(TIPO_ORIGINAL, TIPO, V, VREF, Qcalc, QG, violou_qmax, violou_qmin, iter_bloqueio_reversao, i)

        delta_P = np.zeros(NBAR)
        delta_Q = np.zeros(NBAR)

        for k in range(NBAR):
            if TIPO[k] == 2 or TIPO[k] == 1:  # Slack ou PV local
                QG[k] = Qcalc[k] + QD[k]

        for k in range(NBAR):
            if TIPO[k] == 3:  # PV com controle remoto
                delta_P[k] = Pesp[k] - Pcalc[k]
                delta_Q[k] = (QG[k] - QD[k]) - Qcalc[k]
            elif TIPO[k] == 2:  # Slack
                delta_P[k] = 0
                delta_Q[k] = 0
            elif TIPO[k] == 1:  # PV local
                delta_P[k] = Pesp[k] - Pcalc[k]
                delta_Q[k] = 0
            else:  # PQ
                delta_P[k] = Pesp[k] - Pcalc[k]
                delta_Q[k] = Qesp[k] - Qcalc[k]

        # Cálculo do erro de tensão nas barras controladas
        delta_Vcontrolada = np.zeros(N_controlada)
        for n in range(N_controlada):
            barra_controlada_idx = B_controlada[n] - 1  # índice 0-based
            delta_Vcontrolada[n] = V[barra_controlada_idx] - Vb_controlada[n]

        # Monta vetor delta_Y com resíduos adicionais de controle remoto
        delta_Y = np.concatenate([delta_P, delta_Q, delta_Vcontrolada]).reshape(-1, 1)

        # Vetores de erro separados
        delta_PQ = np.concatenate([delta_P, delta_Q])
        erro_PQ = np.max(np.abs(delta_PQ))
        erro_Vctrl = np.max(np.abs(delta_Vcontrolada)) if N_controlada > 0 else 0.0

        # Define MAX_Y como o pior dos dois erros
        MAX_Y = max(erro_PQ, erro_Vctrl)

        # Debug opcional
        print(f"ΔVcontrolada = {[float(round(x, 6)) for x in delta_Vcontrolada]}")
        print(f"Erro_PQ = {erro_PQ:.6e} | Erro_Vctrl = {erro_Vctrl:.6e} | MAX_Y = {MAX_Y:.6e}")

        if erro_PQ < tolerancia and erro_Vctrl < tol_Vcontrolada:
            convergiu = True
            break


        i += 1  # Incrementa o contador de iterações

        # Matriz Jacobiana
        Jac = montar_matriz_jacobiana(NBAR, V, TETA, Pcalc, Qcalc, G, B, TIPO, B_controle, B_controlada, N_controlada, Cv, Ch)

        # delta_Y deve ser um vetor coluna numpy com dimensão (2*NBAR, 1)
        # Jac é a matriz Jacobiana 2*NBAR x 2*NBAR

        # Resolução do sistema linear: Jac * delta_SOLUCAO = delta_Y
        delta_SOLUCAO = np.linalg.solve(Jac, delta_Y).flatten()

        if i >= 2:  # Só tenta reativar após duas iterações completas
            for c in range(N_controle):
                idx_ctrl = B_controle[c] - 1
                idx_ctrlada = B_controlada[c] - 1

                if TIPO[idx_ctrl] == 3 and TIPO[idx_ctrlada] == 4:
                    continue  # já controlam

                # Só tenta reativar se ambas estão como PQ (controle desativado)
                if TIPO[idx_ctrl] == 0 and TIPO[idx_ctrlada] == 0:
                    dentro_limite = (QG[idx_ctrl] > QN[idx_ctrl] + 1e-4) and (QG[idx_ctrl] < QM[idx_ctrl] - 1e-4)
                    Vctrl = V[idx_ctrlada]
                    Vref = Vb_controlada[c]
                    tensao_desviada = abs(Vctrl - Vref) > 1e-4

                    if dentro_limite and tensao_desviada:
                        TIPO[idx_ctrl] = 3
                        TIPO[idx_ctrlada] = 4
                        Ch[c, 2*NBAR + c] = 0  # reativa a equação
                        print(f"→ Reativação: Barra {B_controle[c]} voltou a controlar a barra {B_controlada[c]}")


        # Atualização dos vetores TETA e V
        TETA += delta_SOLUCAO[0:NBAR]            # primeiros NBAR elementos são delta_TETA
        V += delta_SOLUCAO[NBAR:2*NBAR]             # próximos NBAR elementos são delta_V
        for n in range(N_controle):
            QG[B_controle[n] - 1] += delta_SOLUCAO[2*NBAR + n]
        print("→ V[6] = ", V[5], " | Teta[6] = ", np.degrees(TETA[5]))
        print("→ V[8] = ", V[7], " | Teta[8] = ", np.degrees(TETA[7]))
        
        if Controle_PV:
            for c in range(N_controle):
                idx_controle = B_controle[c] - 1
                idx_controlada = B_controlada[c] - 1

                saturou_qmax = round(QG[idx_controle], 5) >= round(QM[idx_controle], 5)
                saturou_qmin = round(QG[idx_controle], 5) <= round(QN[idx_controle], 5)
                Vctrl = V[idx_controlada]
                Vref = Vb_controlada[c]

                # Caso 1: QG >= Qmax e tensão baixa ⇒ reverte controle
                if saturou_qmax and Vctrl <= Vref:
                    QG[idx_controle] = QM[idx_controle]
                    Ch[c, 2*NBAR + c] = 1e10
                    TIPO[idx_controle] = 0
                    TIPO[idx_controlada] = 0
                    Qesp[idx_controle] = QG[idx_controle] - QD[idx_controle]
                    print(f"→ Reversão para PQ: Barra {B_controle[c]} saturou Qmax e tensão está baixa (V={Vctrl:.4f} ≤ Vref={Vref:.4f})")

                # Caso 2: QG <= Qmin e tensão alta ⇒ reverte controle
                elif saturou_qmin and Vctrl >= Vref:
                    QG[idx_controle] = QN[idx_controle]
                    Ch[c, 2*NBAR + c] = 1e10
                    TIPO[idx_controle] = 0
                    TIPO[idx_controlada] = 0
                    Qesp[idx_controle] = QG[idx_controle] - QD[idx_controle]
                    print(f"→ Reversão para PQ: Barra {B_controle[c]} saturou Qmin e tensão está alta (V={Vctrl:.4f} ≥ Vref={Vref:.4f})")

                # Caso 3: Saturou Q, mas tensão está do “lado bom” ⇒ clipa e mantém controle
                elif QG[idx_controle] > QM[idx_controle] and Vctrl > Vref:
                    QG[idx_controle] = QM[idx_controle]
                    TIPO[idx_controle] = 3
                    TIPO[idx_controlada] = 4

                elif QG[idx_controle] < QN[idx_controle] and Vctrl < Vref:
                    QG[idx_controle] = QN[idx_controle]
                    TIPO[idx_controle] = 3
                    TIPO[idx_controlada] = 4

                # Caso 4: Não violou nada ⇒ mantém controle e equação ativa
                else:
                    Ch[c, 2*NBAR + c] = 0
                    TIPO[idx_controle] = 3
                    TIPO[idx_controlada] = 4


        # Armazena dados da iteração
        relatorio_iteracoes.append({
            'Iteração': i,
            'Mismatch máximo': MAX_Y,
            'Pcalc': Pcalc.copy(),
            'Qcalc': Qcalc.copy(),
            'Pesp': Pesp.copy(),
            'Qesp': Qesp.copy(),
            'delta_P': delta_P.copy(),
            'delta_Q': delta_Q.copy(),
            'delta_V_controlada': delta_Vcontrolada.copy(),
            'V': V.copy(),
            'Teta (graus)': np.degrees(TETA.copy())
        })


        print('-'*80)
        print(f'Iteração {i}')
        print(f'→ Iteração {i}: Mismatch máximo = {MAX_Y:.6e}')
        print('-'*80)
        print(f'{"Nº":<4} {"Tipo":<5} {"V":<8} {"Teta(°)":<10} {"QG":<10} {"Qmin":<10} {"Qmax":<10}')
        for k in range(NBAR):
            print(f'{k+1:<4} {TIPO[k]:<5} {V[k]:<8.4f} {TETA[k]*180/np.pi:<10.2f} {QG[k]*Pbase:<10.4f} {QN[k]*Pbase:<10.4f} {QM[k]*Pbase:<10.4f}')

    for k in range(NLIN):
        K = DE[k] - 1
        M = PARA[k] - 1

        tap = TAP[k] if TAP[k] != 0 else 1.0
        defas = DEFAS[k]

        Vk = V[K]
        Vm = V[M]
        teta_k = TETA[K]
        teta_m = TETA[M]
        gkm = G[K, M]
        bkm = B[K, M]
        bsh = BSH[k]

        delta_km = (teta_k - teta_m) + defas
        delta_mk = (teta_m - teta_k) + defas

        # Fluxo de potência ativa
        fluxo_1 = -(tap * Vk)**2 * gkm + tap * Vk * Vm * (gkm * np.cos(delta_km) + bkm * np.sin(delta_km))
        fluxo_2 = -(tap * Vm)**2 * gkm + tap * Vm * Vk * (gkm * np.cos(delta_mk) + bkm * np.sin(delta_mk))

        # Fluxo de potência reativa
        reat_1 = -(tap**2) * (bsh + bkm) * Vk**2 + tap * Vk * Vm * (bkm * np.cos(delta_km) - gkm * np.sin(delta_km))
        reat_2 = -(tap**2) * (bsh + bkm) * Vm**2 + tap * Vm * Vk * (bkm * np.cos(delta_mk) - gkm * np.sin(delta_mk))

        # Conversão para base real (MW, MVAr)
        fluxo_1 *= Pbase
        fluxo_2 *= Pbase
        reat_1 *= Pbase
        reat_2 *= Pbase

        FLUXO[k, 0] = fluxo_1  # Pkm
        FLUXO[k, 1] = fluxo_2  # Pmk

        if abs(fluxo_1) >= abs(fluxo_2):
            FLUXO[k, 3] = FLUXO[k, 2] - abs(fluxo_1)
            FLUXO[k, 4] = abs(fluxo_1)
            FLUXO[k, 5] = reat_1  # fluxo reativo direto
        else:
            FLUXO[k, 3] = FLUXO[k, 2] - abs(fluxo_2)
            FLUXO[k, 4] = abs(fluxo_2)
            FLUXO[k, 5] = reat_2  # fluxo reativo reverso

    if not Controle_PV:
        TIPO[TIPO == 3] = 1  # Reverte tipo 3 para PV
        TIPO[TIPO == 4] = 0  # Reverte tipo 4 para PQ

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
            print(f'{int(linha[0]):<4} {int(linha[1]):<6} {linha[2]:<8.3f} {linha[3]:<8.2f}{linha[4]:<8.2f} {linha[5]:<8.2f} {linha[6]:<8.2f} {linha[7]:<8.2f}  {linha[8]:<8.2f} {linha[9]:<8.2f}')

        print('='*108)
        print('Fluxo de Potência Ativa entre as Barras')
        print('='*108)
        print(f'\t{"DE":<6}\t{"PARA":<6}\t{"DE-PARA":<10}\t{"PARA-DE":<10}\t{"FLUXO MAX":<10}\t{"FOLGA":<10}')

        # Imprime dados de fluxo: DE, PARA, FLUXO[:,1], FLUXO[:,2], FLUXO[:,3], FLUXO[:,4]
        for i in range(len(DE)):
            print(f'\t{int(DE[i]):<6}\t{int(PARA[i]):<6}\t{FLUXO[i,0]:<10.4f}\t{FLUXO[i,1]:<10.4f}\t{FLUXO[i,2]:<10.4f}\t{FLUXO[i,3]:<10.4f}')

        print('='*108)

        
        nomes = [f"Barra-{i+1:02d}" for i in range(NBAR)]
        relatorio_transformadores(DE, PARA, FLUXO, TAP, LC, nomes_barras=nomes)

    # Expande o relatório e grava em Excel
    dados_expandidos = []
    for item in relatorio_iteracoes:
        for k in range(NBAR):
            dados_expandidos.append({
                'Iteração': item['Iteração'],
                'Barra': k + 1,
                'Mismatch máximo': item['Mismatch máximo'],
                'Pcalc': item['Pcalc'][k],
                'Qcalc': item['Qcalc'][k],
                'Pesp': item['Pesp'][k],
                'Qesp': item['Qesp'][k],
                'ΔP': item['delta_P'][k],
                'ΔQ': item['delta_Q'][k],
                'ΔVcontrol': item['delta_Vcontrolada'][k] if 'delta_Vcontrolada' in item else np.nan,
                'V': item['V'][k],
                'Teta (graus)': item['Teta (graus)'][k]
            })


    df = pd.DataFrame(dados_expandidos)
    df = df.round(5)  # ← arredonda todas as colunas numéricas para 4 casas
    df.to_excel('relatorio_iteracoes_NR.xlsx', index=False)
    print("✔ Relatório salvo como 'relatorio_iteracoes_NR.xlsx'")

    return V, TETA, FLUXO

tensoes, thetas, fluxos = newton_raphson_flow(DBAR, DLIN, Pbase = Pbase, tolerancia = tol)