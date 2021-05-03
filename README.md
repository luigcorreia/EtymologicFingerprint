# Identificação de Clusters Etimológicos no Corpus Brown
### Projeto desenvolvido para disciplina DAMCZA015 - Mineração de Dados, da Universidade Federal do ABC (2019)

Neste projeto exploramos técnicas de agrupamento em documentos da base da Brown University, compilado clássico de documentos de diferentes fontes (gêneros literários, textos jornalísticos, técnicos, discursos informais), utilizando métodos de aprendizado não supervisionado a partir informações de etimologia do léxico empregado, extraídas do Wiktionary com auxílio do projeto Etymological Wordnet. Através da análise das métricas de silhueta e erro quadrático foram identificados 2 grupos básicos, que resultam no maior índice, que podem ser interpretados com o nível de presença de palavras latinas no léxico utilizado.

Link para o artigo discutindo os resultados:
https://bit.ly/3ehfo8K

## Descrição do código-fonte

### datasets.py
Script que utiliza NTLK para processar o Corpus Brown e o dicionário etimológico gerando uma dataset de assinaturas etimológicas para cada documento.
### clusters.py
Script que faz a análise de agrupamento sob o dataset gerado pelo script anterior
