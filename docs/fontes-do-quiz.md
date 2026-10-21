# Fontes por pergunta

Cada pergunta retornada inclui `sources`: identificador, curso, aula, título, versão e trecho do material. A lista geral de aulas em `sources` do quiz permanece disponível, assim como o formato das alternativas e do gabarito.

O modelo recebe identificadores temporários (T1, T2...) e deve informar `source_ids` por pergunta. O serviço recusa referências ausentes ou desconhecidas e solicita uma nova tentativa, respeitando o limite existente. O texto da fonte é obtido do material selecionado pelo servidor, nunca copiado de uma fonte inventada pelo modelo.

A demonstração permite expandir o material de cada questão para revisão. A existência de uma referência não comprova que ela sustenta semanticamente o gabarito: o instrutor deve conferir a pergunta, a explicação e as alternativas. A integração do LMS precisa preservar esse novo campo para mostrar e armazenar as fontes; o serviço de IA não publica avaliações nem altera o acesso dos alunos ao gabarito.
