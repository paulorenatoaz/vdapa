# vdapa

## English Version

**vdapa** (Vulnerability Data Acquisition, Processing and Analysis) is a research project conducted as part of the course ICP719 - Network Security, taught by Professor Daniel S. Menasche at the Instituto de Computação, Universidade Federal do Rio de Janeiro (UFRJ). The project focuses on studying software vulnerabilities using public data sources and aims to answer important questions related to the confirmation of vulnerabilities as formal advisories, based on data from GitHub Security Advisories (GSA), NVD, and EPSS.

### Research Question 
> Can we infer if a given vulnerability will be confirmed by the GitHub Security Advisories (GSA)?

This main question is subdivided into three key topics that guide our investigation:

1. **Time Until Confirmation**  
   Investigating how long it takes from the first public appearance of a vulnerability until it is formally confirmed in the GSA.

2. **GHSA Advisories Without CVE-ID**  
   Understanding why some advisories on GitHub do not have an associated CVE identifier and analyzing differences between advisories with and without CVE.

3. **Improving Time Estimation Using Historical Data**  
   Exploring how historical events related to a vulnerability (commits, issues, pull requests) can improve predictions about when a vulnerability will be confirmed.

### Project Pipeline

To address these questions, the project is divided into four main stages:

- **Data Acquisition**: Collecting raw data from GitHub (via scraping and API), NVD, and EPSS.
- **Data Processing**: Cleaning, merging, and enriching the collected data, creating structured datasets ready for analysis.
- **Data Analysis**: Performing statistical analysis and building models to explore the three topics of RQ1.
- **Reporting**: Generating comprehensive reports with visualizations, summaries, and conclusions for each topic.

This modular pipeline ensures data integrity, reusability, and clarity throughout the research process.

---

## Versão em Português

**vdapa** (Aquisição, Processamento e Análise de Dados de Vulnerabilidades) é um projeto de pesquisa desenvolvido no âmbito da disciplina ICP719 - Segurança em Redes, ministrada pelo Professor Daniel S. Menasché no Instituto de Computação da Universidade Federal do Rio de Janeiro (UFRJ). O projeto tem como foco o estudo de vulnerabilidades de software utilizando dados públicos e tem como objetivo responder perguntas importantes relacionadas à confirmação de vulnerabilidades como avisos formais, baseando-se em dados do GitHub Security Advisories (GSA), NVD e EPSS.

### Pergunta de Pesquisa

> Podemos inferir se uma determinada vulnerabilidade será confirmada pelo GitHub Security Advisories (GSA)?

Essa pergunta principal é subdividida em três tópicos que guiam nossa investigação:

1. **Tempo Até a Confirmação**  
   Investigação sobre quanto tempo leva desde a primeira aparição pública de uma vulnerabilidade até sua confirmação formal no GSA.

2. **Advisories GHSA Sem CVE-ID**  
   Compreensão sobre por que alguns advisories no GitHub não possuem um identificador CVE associado e análise das diferenças entre advisories com e sem CVE.

3. **Melhorando a Estimativa de Tempo com Dados Históricos**  
   Exploração de como eventos históricos relacionados à vulnerabilidade (commits, issues, pull requests) podem melhorar a previsão do momento da confirmação.

### Pipeline do Projeto

Para responder a essas questões, o projeto está dividido em quatro etapas principais:

- **Aquisição de Dados**: Coleta de dados brutos do GitHub (via scraping e API), NVD e EPSS.
- **Processamento de Dados**: Limpeza, junção e enriquecimento dos dados coletados, criando bases estruturadas para análise.
- **Análise de Dados**: Realização de análises estatísticas e construção de modelos para explorar os três tópicos da RQ1.
- **Relatórios**: Geração de relatórios completos com visualizações, resumos e conclusões para cada tópico.

Este pipeline modular assegura integridade dos dados, reutilização e clareza durante todo o processo de pesquisa.

---

For any questions or contributions, feel free to contact the project team.
