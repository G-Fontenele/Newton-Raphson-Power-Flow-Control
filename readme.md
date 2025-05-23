# Newton-Raphson Power Flow with Control

Este repositório contém uma implementação do método de Newton-Raphson para cálculo do fluxo de potência em sistemas elétricos de potência. O código é desenvolvido em Python, permitindo a análise detalhada do estado de tensão, potência ativa e reativa, e inclui a modelagem de controles para barras do tipo PQ, PV e barra slack (swing).

---

## Conteúdo

- Implementação do método de Newton-Raphson para fluxo de potência.
- Modelagem detalhada das barras (PQ, PV, slack) e das linhas de transmissão.
- Critérios de convergência com tolerância configurável.
- Suporte a dados baseados no caso IEEE 14 Barras para testes.
- Estrutura modular para fácil extensão e integração com outros módulos de análise de redes.

---

## Estrutura dos dados

### DBAR (Barras do sistema)

Cada barra é representada por uma lista contendo:
- Número da barra
- Tipo da barra (0 = PQ, 1 = PV, 2 = slack)
- Tensão nominal (pu)
- Ângulo da tensão (graus)
- Potência ativa gerada Pg (pu)
- Potência reativa gerada Qg (pu)
- Potência reativa mínima Qmin (pu)
- Potência reativa máxima Qmax (pu)
- Barra controlada (para controle de tensão, quando aplicável)
- Potência ativa da carga Pl (pu)
- Potência reativa da carga Ql (pu)
- Potência da injeção do gerador Sh (pu), se aplicável
- Área de controle
- Fator de tensão Vf, se aplicável

### DLIN (Linhas de transmissão)

Cada linha é representada por uma lista contendo:
- Barra "de"
- Barra "para"
- Identificador de circuito
- Resistência (ohms ou pu)
- Reatância (ohms ou pu)
- Susceptância da linha (MVAr)
- Tap changer / relação do transformador
- Limites mínimo e máximo do tap changer (se aplicável)
- Ângulo de fase
- Parâmetros adicionais (ex: susceptância em shunt)

---

## Requisitos

- Python 3.7 ou superior
- Bibliotecas:
  - numpy
  - pandas
  - math

---

## Como usar

1. Clone o repositório:

```bash
git clone https://github.com/G-Fontenele/Newton-Raphson-Power-Flow-Control.git
cd Newton-Raphson-Power-Flow-Control
