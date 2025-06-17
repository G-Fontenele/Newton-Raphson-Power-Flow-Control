import math
import numpy as np
import pandas as pd
import copy

# =====================================================================================
# CONFIGURAÇÕES E CONTROLES GLOBAIS
# =====================================================================================
# Ativa/desativa os diferentes tipos de controle
Controle_PV = False      # Controle de limites de potência reativa (PV -> PQ)
Controle_REM = False     # Controle remoto de tensão por gerador (CREM)
Controle_CTAP = True    # Controle de tensão por tap de transformador (CTAP)

# Parâmetros do sistema e da simulação
Pbase = 100             # Potência base (MVA)
tol = 0.0001 / Pbase      # Tolerância de convergência para potências (em pu)
tol_Vcontrolada = 0.5 # Tolerância para as equações de controle de tensão (V_esp - V_calc)
iteracao_max = 100       # Número máximo de iterações

# =====================================================================================
# DADOS DO SISTEMA (IEEE 14 BARRAS - MODIFICADO PARA TESTES)
# =====================================================================================

# -------------------------------------------------------------------------------------
# DBAR: Dados das barras do sistema elétrico
# Formato: [Nº, Tipo, V(pu), Ang(°), Pg(MW), Qg(MVAr), Qmin, Qmax, Barra Remota Crtl, Pl(MW), Ql(MVAr), Shunt(MVAr), Area, Vf]
# -------------------------------------------------------------------------------------
DBAR = [
    [1, 2, 1.060,  0.0,  232.4, -16.3, -999, 999, None,   0.0,   0.0,  0.0, 1, None],
    [2, 1, 1.045,  0.0,   40.0,  44.48, -40.0, 50.0, None,  21.7,  12.7,  0.0, 1, None],
    [3, 1, 1.010,  0.0,    0.0,  25.67,   0.0, 40.0, None,  94.2,  19.0,  0.0, 1, None],
    [4, 0, 1.017,  0.0,    0.0,    0.0, -999, 999, None,  47.8,  -3.9,  0.0, 1, None],
    [5, 0, 1.020,  0.0,    0.0,    0.0, -999, 999, None,   7.6,   1.6,  0.0, 1, None],
    [6, 1, 1.070,  0.0,    0.0,  11.68,  -6.0, 24.0, 12,  11.2,   7.5,  0.0, 1, None],
    [7, 0, 1.000,  0.0,    0.0,    0.0, -999, 999, None,   0.0,   0.0,  0.0, 1, None],
    [8, 1, 1.090,  0.0,    0.0,  16.68,  -6.0, 24.0, 7,   0.0,   0.0,  0.0, 1, None],
    [9, 0, 1.060,  0.0,    0.0,    0.0, -999, 999, None,  29.5,  16.6, 19.0, 1, None],
    [10,0, 1.054,  0.0,    0.0,    0.0, -999, 999, None,   9.0,   5.8,  0.0, 1, None],
    [11,0, 1.059,  0.0,    0.0,    0.0, -999, 999, None,   3.5,   1.8,  0.0, 1, None],
    [12,0, 1.070,  0.0,    0.0,    0.0, -999, 999, None,   6.1,   1.6,  0.0, 1, None],
    [13,0, 1.051,  0.0,    0.0,    0.0, -999, 999, None,  13.5,   5.8,  0.0, 1, None],
    [14,0, 1.038,  0.0,    0.0,    0.0, -999, 999, None,  14.9,   5.0,  0.0, 1, None]
]

# -------------------------------------------------------------------------------------
# DLIN: Dados das linhas de transmissão
# Formato: [De, Para, Circ, R(%), X(%), B(MVAr), Tap, Tmin, Tmax, Fase(°), Barra Tap Crtl, Cn, Ce, Ns]
# -------------------------------------------------------------------------------------
DLIN = [
    [1, 2, 1, 1.938, 5.917, 5.28, 1.0, None, None, 0.0, None, None, None, None],
    [1, 5, 1, 5.403, 22.304, 4.92, 1.0, None, None, 0.0, None, None, None, None],
    [2, 3, 1, 4.699, 19.797, 4.38, 1.0, None, None, 0.0, None, None, None, None],
    [2, 4, 1, 5.811, 17.632, 3.40, 1.0, None, None, 0.0, None, None, None, None],
    [2, 5, 1, 5.695, 17.388, 3.46, 1.0, None, None, 0.0, None, None, None, None],
    [3, 4, 1, 6.701, 17.103, 1.28, 1.0, None, None, 0.0, None, None, None, None],
    [4, 5, 1, 1.335, 4.211, 0.0, 1.0, None, None, 0.0, None, None, None, None],
    [4, 7, 1, 0.0, 20.912, 0.0, 0.978, None, None, 0.0, None, None, None, None],
    [4, 9, 1, 0.0, 55.618, 0.0, 0.969, 0.6, 1.4, 0.0, 9, None, None, None],
    [5, 6, 1, 0.0, 25.202, 0.0, 0.9126, 0.6, 1.4, 0.0, 5, None, None, None],
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

# =====================================================================================
# FUNÇÕES AUXILIARES
# =====================================================================================

def montar_ybarra(NBAR, NLIN, DE, PARA, R, X, BSH, TAP, DEFAS, SHUNT):
    """
    Monta a matriz de admitância nodal (Ybarra) do sistema.
    """
    Ybarra = np.zeros((NBAR, NBAR), dtype=complex)
    Ykm = 1 / (R + 1j * X)
    Bsh = 1j * BSH

    for k in range(NLIN):
        de_idx = int(DE[k]) - 1
        para_idx = int(PARA[k]) - 1
        tap_k = TAP[k]
        defas_k = DEFAS[k]
        
        Ybarra[de_idx, de_idx] += Ykm[k] / (tap_k**2) + Bsh[k]
        Ybarra[para_idx, para_idx] += Ykm[k] + Bsh[k]
        Ybarra[de_idx, para_idx] -= Ykm[k] / tap_k * np.exp(-1j * defas_k)
        Ybarra[para_idx, de_idx] -= Ykm[k] / tap_k * np.exp(1j * defas_k)
    
    for k in range(NBAR):
        Ybarra[k, k] += 1j * SHUNT[k]

    return Ybarra, np.real(Ybarra), np.imag(Ybarra)


def montar_matriz_jacobiana_aumentada(NBAR, V, TETA, Pcalc, Qcalc, G, B, TIPO, controles, Ch):
    """
    Monta a matriz Jacobiana aumentada, incorporando as equações de todos os
    controles ativos (CREM e CTAP). Agora usa a matriz Ch para o controle REM.
    """
    rem_info = controles['rem']
    tap_info = controles['tap']
    
    N_REM = rem_info['n_controle']
    N_CTAP = tap_info['n_controle']
    dim_total = 2 * NBAR + N_REM + N_CTAP
    
    Jac = np.zeros((dim_total, dim_total))
    
    H, N, M, L = np.zeros((NBAR, NBAR)), np.zeros((NBAR, NBAR)), np.zeros((NBAR, NBAR)), np.zeros((NBAR, NBAR))

    for k1 in range(NBAR):
        for k2 in range(NBAR):
            if k1 == k2:
                H[k1, k1] = -Qcalc[k1] - V[k1]**2 * B[k1, k1]
                N[k1, k1] = (Pcalc[k1] + V[k1]**2 * G[k1, k1]) / V[k1]
                M[k1, k1] = Pcalc[k1] - V[k1]**2 * G[k1, k1]
                L[k1, k1] = (Qcalc[k1] - V[k1]**2 * B[k1, k1]) / V[k1]
            else:
                d_theta = TETA[k1] - TETA[k2]
                H[k1, k2] = V[k1] * V[k2] * (G[k1, k2] * np.sin(d_theta) - B[k1, k2] * np.cos(d_theta))
                N[k1, k2] = V[k1] * (G[k1, k2] * np.cos(d_theta) + B[k1, k2] * np.sin(d_theta))
                M[k1, k2] = -V[k1] * V[k2] * (G[k1, k2] * np.cos(d_theta) + B[k1, k2] * np.sin(d_theta))
                L[k1, k2] = V[k1] * (G[k1, k2] * np.sin(d_theta) - B[k1, k2] * np.cos(d_theta))

    Jac[0:NBAR, 0:NBAR] = H
    Jac[0:NBAR, NBAR:2*NBAR] = N
    Jac[NBAR:2*NBAR, 0:NBAR] = M
    Jac[NBAR:2*NBAR, NBAR:2*NBAR] = L

    # Aplica "Big Number" para fixar variáveis de barras Slack (Tipo 2) e PV (Tipo 1)
    for k in range(NBAR):
        if TIPO[k] == 2:
            Jac[k, k] = 1e10
            Jac[NBAR + k, NBAR + k] = 1e10
        elif TIPO[k] == 1:
            Jac[NBAR + k, NBAR + k] = 1e10

    # --- INÍCIO DA ATUALIZAÇÃO ---

    if Controle_REM and N_REM > 0:
        # 1. Acopla a variável de estado dQg à equação de potência reativa da controladora.
        for i in range(N_REM):
            idx_controladora = rem_info['controladoras'][i] - 1
            idx_var_qg = 2 * NBAR + i  # Coluna correspondente a dQg
            # Este termo (-1) representa a derivada d(Qesp - Qcalc)/dQg, seguindo a convenção J = -dF/dx
            Jac[NBAR + idx_controladora, idx_var_qg] = -1.0

        # 2. Adiciona as equações de controle de tensão (e seus interruptores) da matriz Ch.
        jac_rem_start_row = 2 * NBAR
        jac_rem_end_row = 2 * NBAR + N_REM
        Jac[jac_rem_start_row:jac_rem_end_row, :] = Ch
    
    # --- FIM DA ATUALIZAÇÃO ---

    if Controle_CTAP and N_CTAP > 0:
        for i in range(N_CTAP):
            ramal_idx = tap_info['indices_ramos'][i]
            k_idx = tap_info['de_para'][ramal_idx][0] - 1
            m_idx = tap_info['de_para'][ramal_idx][1] - 1
            idx_controlada = tap_info['controladas'][i] - 1
            
            tap = tap_info['taps'][ramal_idx]
            
            # As derivadas parciais aqui podem precisar de revisão.
            # Usando uma aproximação onde G e B são constantes (sem tap), para simplificar.
            # O ideal é recalcular G e B em função do tap a cada iteração.
            # No entanto, mantendo a sua formulação original:
            # ykm = 1 / (0.01*DLIN[ramal_idx][3] + 1j*0.01*DLIN[ramal_idx][4]) # r e x em pu
            # gkm_val, bkm_val = np.real(ykm), np.imag(ykm)
            
            # Dados da linha
            gkm = G[k_idx, m_idx]
            bkm = B[k_idx, m_idx]

            idx_var_tap = 2 * NBAR + N_REM + i
            
            # dPk_da = -2*V[k_idx]**2*gkm_val/tap**3 + V[k_idx]*V[m_idx]/tap**2 * (gkm_val*np.cos(TETA[k_idx]-TETA[m_idx]) + bkm_val*np.sin(TETA[k_idx]-TETA[m_idx]))
            # dQk_da = 2*V[k_idx]**2*bkm_val/tap**3 - V[k_idx]*V[m_idx]/tap**2 * (gkm_val*np.sin(TETA[k_idx]-TETA[m_idx]) - bkm_val*np.cos(TETA[k_idx]-TETA[m_idx]))
            # dPm_da = -V[k_idx]*V[m_idx]/tap**2 * (gkm_val*np.cos(TETA[k_idx]-TETA[m_idx]) + bkm_val*np.sin(TETA[k_idx]-TETA[m_idx]))
            # dQm_da = -V[k_idx]*V[m_idx]/tap**2 * (gkm_val*np.sin(TETA[k_idx]-TETA[m_idx]) - bkm_val*np.cos(TETA[k_idx]-TETA[m_idx]))

            # Derivadas das potências em relação ao TAP
            Vk = V[k_idx]
            Vm = V[m_idx]
            teta_k = TETA[k_idx]
            teta_m = TETA[m_idx]

            dPk_da = 2 * tap * Vk**2 * gkm - Vk * Vm * (gkm * np.cos(teta_k - teta_m) + bkm * np.sin(teta_k - teta_m))
            dPm_da = -Vk * Vm * (gkm * np.cos(teta_m - teta_k) - bkm * np.sin(teta_m - teta_k))
            dQk_da = -2 * tap * Vk**2 * bkm + Vk * Vm * (bkm * np.cos(teta_k - teta_m) - gkm * np.sin(teta_k - teta_m))
            dQm_da = Vk * Vm * (bkm * np.cos(teta_m - teta_k) + gkm * np.sin(teta_m - teta_k))

            Jac[k_idx, idx_var_tap] = dPk_da
            Jac[NBAR + k_idx, idx_var_tap] = dQk_da
            Jac[m_idx, idx_var_tap] = dPm_da
            Jac[NBAR + m_idx, idx_var_tap] = dQm_da
            
            # Equação de controle do TAP V_esp - V_calc = 0
            # A derivada d(V_esp - V_calc) / dV_calc = -1. Com J=-dF/dx, o termo é +1.
            Jac[2*NBAR + N_REM + i, NBAR + idx_controlada] = 1.0

    return Jac

def verificar_limites_Q(PV, Qcalc, QM, QN, QG, TIPO, Pbase, i, violacoes, QD):
    """Verifica se as barras PV violaram seus limites de geração de reativos."""
    for idx in PV:
        q_gerado = Qcalc[idx] + QD[idx]
        if q_gerado > QM[idx]:
            print(f"Iter {i}: Barra {idx+1} VIOLOU Qmax. Virou PQ. (Qgerado={q_gerado*Pbase:.2f} > Qmax={QM[idx]*Pbase:.2f})")
            TIPO[idx] = 0
            QG[idx] = QM[idx]
            violacoes[idx] = {'tipo': 'qmax', 'iter': i}
        elif q_gerado < QN[idx]:
            print(f"Iter {i}: Barra {idx+1} VIOLOU Qmin. Virou PQ. (Qgerado={q_gerado*Pbase:.2f} < Qmin={QN[idx]*Pbase:.2f})")
            TIPO[idx] = 0
            QG[idx] = QN[idx]
            violacoes[idx] = {'tipo': 'qmin', 'iter': i}
    return QG, TIPO, violacoes

def reverter_tipo_barras(TIPO_ORIGINAL, TIPO, V, V_ESP, violacoes, i):
    """Tenta reverter barras que viraram PQ de volta para PV."""
    for idx, viol in violacoes.items():
        if TIPO[idx] == 0 and TIPO_ORIGINAL[idx] == 1 and i > viol['iter']:
            delta_V = V_ESP[idx] - V[idx]
            reverter = False
            if viol['tipo'] == 'qmax' and delta_V < -1e-5:
                reverter = True
            elif viol['tipo'] == 'qmin' and delta_V > 1e-5:
                reverter = True

            if reverter:
                print(f"Iter {i}: Barra {idx+1} REVERTEU para PV. (dV = {delta_V:.4f})")
                TIPO[idx] = 1
                V[idx] = V_ESP[idx]
                violacoes[idx]['tipo'] = None
    return TIPO, V, violacoes

def relatorio_barras(NBAR, TIPO, V, TETA, Pcalc, Qcalc, PD, QD, QN, QM, Pbase):
    """Imprime um relatório final detalhado do estado das barras."""
    print('='*110)
    print(f'{"Nº":>3} {"Tipo":>4} {"V (pu)":>8} {"Ang(°)":>9} {"PG (MW)":>11} {"QG (MVAr)":>12} {"PL (MW)":>11} {"QL (MVAr)":>12} {"Qmin":>10} {"Qmax":>10}')
    print('-'*110)

    pg_final = (Pcalc + PD) * Pbase
    qg_final = (Qcalc + QD) * Pbase
    
    for k in range(NBAR):
        print(f"{k+1:>3} {TIPO[k]:>4} {V[k]:>8.4f} {np.degrees(TETA[k]):>9.3f} "
              f"{pg_final[k]:>11.2f} {qg_final[k]:>12.2f} "
              f"{PD[k]*Pbase:>11.2f} {QD[k]*Pbase:>12.2f} "
              f"{QN[k]*Pbase:>10.2f} {QM[k]*Pbase:>10.2f}")
    print('='*110)

def controle_tensao_reversivel(rem_info, QG, QM, QN, V, Vb_controlada, Qesp, QD, TIPO, Pbase, Ch, NBAR):
    for c in range(rem_info['n_controle']):
        idx_controle = rem_info['controladoras'][c] - 1
        idx_controlada = rem_info['controladas'][c] - 1

        saturou_qmax = round(QG[idx_controle], 5) >= round(QM[idx_controle], 5)
        saturou_qmin = round(QG[idx_controle], 5) <= round(QN[idx_controle], 5)
        Vctrl = V[idx_controlada]
        Vref = Vb_controlada[c]

        if saturou_qmax and Vctrl <= Vref:
            QG[idx_controle] = QM[idx_controle]
            Ch[c, 2*NBAR + c] = 1e10
            TIPO[idx_controle] = 0
            TIPO[idx_controlada] = 0
            Qesp[idx_controle] = QG[idx_controle] - QD[idx_controle]
            print(f"→ Reversão para PQ: Barra {idx_controle+1} saturou Qmax e V={Vctrl:.4f} ≤ Vref={Vref:.4f}")

        elif saturou_qmin and Vctrl >= Vref:
            QG[idx_controle] = QN[idx_controle]
            Ch[c, 2*NBAR + c] = 1e10
            TIPO[idx_controle] = 0
            TIPO[idx_controlada] = 0
            Qesp[idx_controle] = QG[idx_controle] - QD[idx_controle]
            print(f"→ Reversão para PQ: Barra {idx_controle+1} saturou Qmin e V={Vctrl:.4f} ≥ Vref={Vref:.4f}")

        elif QG[idx_controle] > QM[idx_controle] and Vctrl > Vref:
            QG[idx_controle] = QM[idx_controle]
            TIPO[idx_controle] = 3
            TIPO[idx_controlada] = 4

        elif QG[idx_controle] < QN[idx_controle] and Vctrl < Vref:
            QG[idx_controle] = QN[idx_controle]
            TIPO[idx_controle] = 3
            TIPO[idx_controlada] = 4

        else:
            TIPO[idx_controle] = 3
            TIPO[idx_controlada] = 4

def relatorio_transformadores(NLIN, DE, PARA, FLUXO, TAP, LC, Pbase):
    """Imprime um relatório dos fluxos e status dos transformadores."""
    print("\nRELATÓRIO DE FLUXO NOS RAMOS E STATUS DOS TRANSFORMADORES")
    print("="*100)
    print(f'{"De":>4} {"Para":>6} {"Pkm (MW)":>12} {"Qkm (MVAr)":>14} {"Pmk (MW)":>12} {"Qmk (MVAr)":>14} {"Tap":>8} {"Tipo":>6}')
    print("-"*100)

    for k in range(NLIN):
        pkm = FLUXO[k, 0] * Pbase
        qkm = FLUXO[k, 1] * Pbase
        pmk = FLUXO[k, 2] * Pbase
        qmk = FLUXO[k, 3] * Pbase
        
        tipo_tap = "Fixo"
        if LC[k] is not None:
            tipo_tap = "Ctrl"

        print(f"{int(DE[k]):>4} -> {int(PARA[k]):<4} {pkm:>12.2f} {qkm:>14.2f} {pmk:>12.2f} {qmk:>14.2f} {TAP[k]:>8.4f} {tipo_tap:>6}")
    print("="*100)

def imprimir_matriz(matriz, nome, limiar=1e-5, arquivo=None):
    """Imprime uma matriz formatada com destaque para valores significativos"""
    n_linhas, n_cols = matriz.shape
    output = f"\n{'-'*50}\n{nome} ({n_linhas}x{n_cols}):\n{'-'*50}\n"
    
    for i in range(n_linhas):
        linha = []
        for j in range(n_cols):
            val = matriz[i,j]
            if abs(val) > limiar:
                linha.append(f"{val:>9.3f}")
            else:
                linha.append(f"{' ':>9}")
        output += " ".join(linha) + "\n"
    
    if arquivo:
        with open(arquivo, 'a') as f:
            f.write(output + '\n')
    else:
        print(output)

def salvar_matrizes_iteracao(i, Ybarra, Jac, delta_Y, V, TETA, QG, TAP, controles):
    """Salva as matrizes relevantes da iteração i em um arquivo Excel para facilitar visualização"""

    # Cria um dicionário com DataFrames a serem exportados
    writer = pd.ExcelWriter(f'debug_iteracao_{i}.xlsx', engine='xlsxwriter')

    # Matriz Ybarra
    pd.DataFrame(np.round(np.real(Ybarra), 6)).to_excel(writer, sheet_name='Ybarra_real', index=False, header=False)
    pd.DataFrame(np.round(np.imag(Ybarra), 6)).to_excel(writer, sheet_name='Ybarra_imag', index=False, header=False)

    # Jacobiana
    pd.DataFrame(np.round(Jac, 6)).to_excel(writer, sheet_name='Jacobiana', index=False, header=False)

    # Vetor delta_Y
    pd.DataFrame(np.round(delta_Y, 6)).to_excel(writer, sheet_name='delta_Y', index=False, header=False)

    # Tensões e ângulos
    df_estado = pd.DataFrame({
        'V (pu)': V,
        'Teta (graus)': np.degrees(TETA),
        'QG (pu)': QG
    })
    df_estado.to_excel(writer, sheet_name='Estado', index=False)

    # TAPs (se houver controle)
    if Controle_CTAP and 'tap' in controles:
        try:
            taps_info = {
                f"TAP_{idx+1}": TAP[idx]
                for idx in controles['tap']['indices_ramos']
            }
            pd.DataFrame(taps_info, index=[0]).to_excel(writer, sheet_name='TAPs', index=False)
        except Exception as e:
            print(f"[!] Erro ao salvar TAPs: {e}")

    writer.close()


# =====================================================================================
# FUNÇÃO PRINCIPAL: NEWTON-RAPHSON INTEGRADO
# =====================================================================================

def newton_raphson_integrado(DBAR, DLIN, Pbase, tolerancia, tol_Vcontrolada, iteracao_max, printar_relatorio=True, debug=True):
    """
    Executa o fluxo de potência pelo método de Newton-Raphson com controles integrados.
    A impressão dos relatórios no console é condicional.
    """
    # 1. Extração e Inicialização de Dados
    NBAR = len(DBAR)
    NLIN = len(DLIN)
    
    TIPO_ORIGINAL = np.array([row[1] for row in DBAR])
    TIPO = TIPO_ORIGINAL.copy()
    V = np.array([row[2] for row in DBAR])
    V_ESP = V.copy()
    TETA = np.radians([row[3] for row in DBAR])
    PG = np.array([row[4] for row in DBAR]) / Pbase
    QG = np.array([row[5] for row in DBAR]) / Pbase
    QN = np.array([row[6] for row in DBAR]) / Pbase
    QM = np.array([row[7] for row in DBAR]) / Pbase
    PD = np.array([row[9] for row in DBAR]) / Pbase
    QD = np.array([row[10] for row in DBAR]) / Pbase
    SHUNT = np.array([row[11] for row in DBAR]) / Pbase
    
    DE = np.array([row[0] for row in DLIN])
    PARA = np.array([row[1] for row in DLIN])
    R = np.array([row[3] for row in DLIN]) / 100
    X = np.array([row[4] for row in DLIN]) / 100
    BSH = np.array([row[5] for row in DLIN]) / (2 * Pbase)
    TAP_ORIGINAL = np.array([row[6] for row in DLIN])
    TAP = TAP_ORIGINAL.copy()
    DEFAS = np.radians([row[9] for row in DLIN])
    
    # 2. Identificação e Configuração dos Controles
    rem_info = {'controladoras': [], 'controladas': [], 'v_esp': [], 'n_controle': 0}
    if Controle_REM:
        for i, barra in enumerate(DBAR):
            if barra[8] is not None:
                rem_info['controladoras'].append(barra[0])
                rem_info['controladas'].append(barra[8])
                rem_info['v_esp'].append(V_ESP[barra[8]-1])
                TIPO[i] = 3
        rem_info['n_controle'] = len(rem_info['controladoras'])

    tap_info = {'indices_ramos': [], 'controladas': [], 'v_esp': [], 'n_controle': 0,
                'taps': TAP, 'de_para': {i: (DE[i], PARA[i]) for i in range(NLIN)}}
    if Controle_CTAP:
        for i, linha in enumerate(DLIN):
            if linha[10] is not None:
                tap_info['indices_ramos'].append(i)
                tap_info['controladas'].append(linha[10])
                tap_info['v_esp'].append(V_ESP[linha[10]-1])
        tap_info['n_controle'] = len(tap_info['indices_ramos'])

    controles = {'rem': rem_info, 'tap': tap_info}
    violacoes_q = {}
    n_rem = rem_info['n_controle']
    n_tap = tap_info['n_controle']
    dim_total = 2 * NBAR + n_rem + n_tap
    Ch = np.zeros((n_rem, dim_total))

    for c in range(n_rem):
        idx_controlada = rem_info['controladas'][c] - 1
        Ch[c, NBAR + idx_controlada] = 1.0

    # 3. Loop Iterativo do Newton-Raphson
    i = 0
    convergiu = False
    relatorio_iteracoes = []
    erro_pot = float('inf')
    erro_ctrl = float('inf')
    Pcalc, Qcalc = None, None

    if printar_relatorio:
        print("Iniciando processo iterativo do Fluxo de Potência Newton-Raphson...")

    while not convergiu and i < iteracao_max:
        i += 1
        if printar_relatorio:
            print(f"\n---------- Iteração {i} ----------")

        Ybarra, G, B = montar_ybarra(NBAR, NLIN, DE, PARA, R, X, BSH, TAP, DEFAS, SHUNT)
        Vret = V * np.exp(1j * TETA)
        Iinj = Ybarra @ Vret
        Sinj = Vret * np.conj(Iinj)
        Pcalc, Qcalc = np.real(Sinj), np.imag(Sinj)

        if Controle_PV and i > 1:
            PV_atuais = np.where(TIPO == 1)[0]
            QG, TIPO, violacoes_q = verificar_limites_Q(PV_atuais, Qcalc, QM, QN, QG, TIPO, Pbase, i, violacoes_q, QD)
            TIPO, V, violacoes_q = reverter_tipo_barras(TIPO_ORIGINAL, TIPO, V, V_ESP, violacoes_q, i)

        Pesp = PG - PD
        Qesp = QG - QD
        delta_P = Pesp - Pcalc
        delta_Q = Qesp - Qcalc

        for k in range(NBAR):
            if TIPO[k] == 2:
                delta_P[k], delta_Q[k] = 0, 0
            elif TIPO[k] == 1 or TIPO[k] == 3:
                delta_Q[k] = 0

        delta_V_rem = np.zeros(rem_info['n_controle'])
        if Controle_REM:
            for n in range(rem_info['n_controle']):
                delta_V_rem[n] = rem_info['v_esp'][n] - V[rem_info['controladas'][n] - 1]

        delta_V_tap = np.zeros(tap_info['n_controle'])
        if Controle_CTAP:
            for n in range(tap_info['n_controle']):
                delta_V_tap[n] = tap_info['v_esp'][n] - V[tap_info['controladas'][n] - 1]

        delta_Y = np.concatenate([delta_P, delta_Q, delta_V_rem, delta_V_tap]).reshape(-1, 1)

        erro_pot = np.max(np.abs(np.concatenate([delta_P, delta_Q])))
        erro_ctrl = np.max(np.abs(np.concatenate([delta_V_rem, delta_V_tap]))) if (rem_info['n_controle'] + tap_info['n_controle']) > 0 else 0
        
        if printar_relatorio:
            print(f"Mismatch Potência: {erro_pot:.6f} | Mismatch Controle: {erro_ctrl:.6f}")

        if erro_pot < tolerancia and erro_ctrl < tol_Vcontrolada:
            convergiu = True
            break
   
        Jac = montar_matriz_jacobiana_aumentada(NBAR, V, TETA, Pcalc, Qcalc, G, B, TIPO, controles, Ch)

        if debug and i <= 5:  # Salvar apenas as primeiras iterações para análise
            salvar_matrizes_iteracao(i, Ybarra, Jac, delta_Y, V, TETA, QG, TAP, controles)
        
        try:
            delta_SOLUCAO = np.linalg.solve(Jac, delta_Y).flatten()
        except np.linalg.LinAlgError as e:
            if debug:
                # Salvar estado atual para análise pós-falha
                np.savez(f"erro_iteracao_{i}.npz", 
                         Jac=Jac, delta_Y=delta_Y, V=V, TETA=TETA,
                         QG=QG, TAP=TAP, Pcalc=Pcalc, Qcalc=Qcalc)
            if printar_relatorio:
                print(f"\nERRO: Matriz Jacobiana singular na iteração {i}. Detalhes:")
                print(f" - Condicionamento: {np.linalg.cond(Jac):.2e}")
                print(f" - Rank: {np.linalg.matrix_rank(Jac)} de {Jac.shape[0]}")
                print(f" - Det: {np.linalg.det(Jac):.2e}")
            return None, None, None

        try:
            delta_SOLUCAO = np.linalg.solve(Jac, delta_Y).flatten()
        except np.linalg.LinAlgError:
            if printar_relatorio:
                print("\nERRO: Matriz Jacobiana singular. O sistema divergiu.")
            return None, None, None

        TETA += delta_SOLUCAO[0:NBAR]
        V += delta_SOLUCAO[NBAR:2*NBAR]

        if Controle_REM and rem_info['n_controle'] > 0:
            delta_QG = delta_SOLUCAO[2*NBAR : 2*NBAR + rem_info['n_controle']]
            for n in range(rem_info['n_controle']):
                QG[rem_info['controladoras'][n] - 1] += delta_QG[n]
        # controle_tensao_reversivel(rem_info, QG, QM, QN, V, rem_info['v_esp'], Qesp, QD, TIPO, Pbase, Ch, NBAR)        

        if Controle_CTAP and tap_info['n_controle'] > 0:
            delta_TAP = delta_SOLUCAO[2*NBAR + rem_info['n_controle']:]
            print(f"Resíduo de tensão para o TAP: {delta_TAP}")
            for n in range(tap_info['n_controle']):
                tap_atualizado = delta_TAP[n] + TAP[tap_info['indices_ramos'][n]]
                
                idx_controlada = tap_info['controladas'][n] - 1
                v_calculado = V[idx_controlada]
                v_especificado = tap_info['v_esp'][n] # Ou de onde vier seu V_esp
                print(f"V_calculado na barra {idx_controlada+1}: {v_calculado:.5f}")
                print(f"V_especificado: {v_especificado:.5f}")
                print(f"Diferença (V_esp - V_calc): {v_especificado - v_calculado:.5f}")

                # imposição de limite forçada
                if tap_atualizado < 0.8:
                    TAP[tap_info['indices_ramos'][n]] = 0.8
                elif tap_atualizado > 1.2:
                    TAP[tap_info['indices_ramos'][n]] = 1.2
                else: TAP[tap_info['indices_ramos'][n]] = tap_atualizado
        
        print(TAP)

        relatorio_iteracoes.append({
            'Iteração': i, 'V': V.copy(), 'Teta (graus)': np.degrees(TETA.copy()),
            'delta_P': delta_P.copy(), 'delta_Q': delta_Q.copy(), 'delta_V_rem': delta_V_rem.copy(),
            'delta_V_tap': delta_V_tap.copy(), 'TAP': TAP.copy(), 'QG': QG.copy()
        })

    # 4. Pós-processamento e Relatórios Finais
    FLUXO = np.zeros((NLIN, 4))
    if convergiu and Pcalc is not None:
        Vret = V * np.exp(1j * TETA)
        for k in range(NLIN):
            de = DE[k] - 1
            para = PARA[k] - 1
            tap = TAP[k]
            ykm = 1 / (R[k] + 1j*X[k])
            bsh_k = 1j * BSH[k]
            
            I_km = (Vret[de]/tap - Vret[para]) * (ykm / tap) + Vret[de]*bsh_k
            I_mk = (Vret[para] - Vret[de]*tap) * ykm + Vret[para]*bsh_k
            S_km = Vret[de] * np.conj(I_km)
            S_mk = Vret[para] * np.conj(I_mk)

            FLUXO[k, 0] = np.real(S_km)
            FLUXO[k, 1] = np.imag(S_km)
            FLUXO[k, 2] = np.real(S_mk)
            FLUXO[k, 3] = np.imag(S_mk)

    if printar_relatorio:
        if not convergiu:
            print('\nX--------------------------X  F L U X O  D E  C A R G A  X---------------------------X')
            print(f'O caso DIVERGIU após {i} iterações.')
            print('X---------------------------------------------------------------------------------X')
        else:
            mismatch_final = max(erro_pot, erro_ctrl)
            print('\nX--------------------------X  F L U X O  D E  C A R G A  X---------------------------X')
            print(f'→ Convergência obtida em {i} iterações')
            print(f'→ Resíduo Máximo: {mismatch_final:.6g} < Tolerância de {max(tolerancia, tol_Vcontrolada):.6f}')
            print('X---------------------------------------------------------------------------------X')
            
            relatorio_barras(NBAR, TIPO, V, TETA, Pcalc, Qcalc, PD, QD, QN, QM, Pbase)
            print('X---------------------------------------------------------------------------------X')
            relatorio_transformadores(NLIN, DE, PARA, FLUXO, TAP, [lin[10] for lin in DLIN], Pbase)

        df = pd.DataFrame(relatorio_iteracoes)
        df = df.round(5)
        df.to_excel('relatorio_iteracoes_NR_integrado.xlsx', index=False)
        print("\n✔ Relatório de iterações salvo como 'relatorio_iteracoes_NR_integrado.xlsx'")

    return V, TETA, FLUXO


# =====================================================================================
# EXECUÇÃO DO SCRIPT
# =====================================================================================
if __name__ == '__main__':
    tensoes, angulos, fluxos = newton_raphson_integrado(DBAR, DLIN, Pbase, tol, tol_Vcontrolada, iteracao_max=20, debug=False)