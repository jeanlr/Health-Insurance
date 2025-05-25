# 🏥 Previsão de Custos com Plano de Saúde

Este projeto analisa os principais fatores que impactam os custos individuais de planos de saúde em uma empresa, utilizando técnicas de análise exploratória de dados (EDA) e modelos preditivos de machine learning para prever e identificar os principais fatores de risco e custos.

## 🎯 Objetivo

Desenvolver análises e modelos que identifiquem características dos colaboradores relacionadas a maiores custos assistenciais, permitindo prever o custo individual do plano de saúde e apoiar ações estratégicas.

## 📊 Dataset

- **Total de registros**: 1.338 colaboradores
- **Completude dos dados**: Nenhum valor nulo em nenhuma feature (0% de missing)

### Variáveis numéricas contínuas:

- `IMC`: Média de 30.66, variando de 15.96 a 53.13
- `Idade`: Média de 39.2 anos
- `Custo_Saude` (target): Média de $1.327, variando de $112 a $6.377

### Variáveis categóricas relevantes:

- `IMC_Classificacao`: 6 categorias (de "Abaixo do Peso" a "Obesidade Grau III")
- `Fumante`: Binário original (Sim/Não) e codificado (`Fumante_Sim`=0/1)
- `Região`: 4 categorias geográficas
- `Sexo`: Distribuição balanceada (2 categorias)

### Features criadas com engenharia de atributos:

- `ordem_IMC`: Classificação ordinal do IMC (1-6)
- `IMC_Classificacao_e_Fumante`: Combinação categórica (12 valores únicos)
- `ordem_IMC_e_Fumante_Sim`: Interação numérica entre IMC e tabagismo (7 valores únicos)

## 🔍 Análise Exploratória (EDA)

- **Tabagismo vs. Custo**: Alta correlação (R² = 0.6198). Fumantes têm custos significativamente maiores.
- **ordem_IMC_e_Fumante_Sim vs. Custo**: Melhor poder explicativo (R² = 0.7343). A combinação de IMC elevado e tabagismo é o fator mais crítico para custos altos.

## 🤖 Inferência e Modelagem Preditiva

### Regressão Linear Múltipla

- **Fórmula da Regressão**:
-- **Custo_Saude**: -219.64 + 665.81 × ordem_IMC_e_Fumante_Sim + 26.90 × Idade
- **R²**: 0.832 (83.2% da variação nos custos de saúde explicada pelas variáveis selecionadas)

### Random Forest Regressor

- **R²**: Aproximadamente 0.87 (87% da variação nos custos de saúde explicada pelas variáveis selecionadas)
- **MAE**: Erro médio absoluto de $250

## ✅ Conclusão

A análise demonstrou que é possível prever os custos individuais de planos de saúde com alta precisão, permitindo que empresas adotem medidas proativas para gestão de custos e bem-estar dos colaboradores, baseando-se em insights obtidos através de dados.

## 📁 Notebooks

- [Análise Exploratória (EDA)](https://github.com/jeanrodovalho16/previsao-custos-saude/blob/main/eda.ipynb)
- [Modelagem Preditiva](https://github.com/jeanrodovalho16/previsao-custos-saude/blob/main/modelagem.ipynb)

## 📝 Artigo Completo

Para uma descrição detalhada do projeto, acesse o artigo no Medium:

[Previsão de custos com plano de saúde: Entendendo os fatores de risco em uma empresa](https://medium.com/@jeanrodovalho16/previs%C3%A3o-de-custos-com-plano-de-sa%C3%BAde-entendendo-os-fatores-de-risco-em-uma-empresa-309a49a022a4)
