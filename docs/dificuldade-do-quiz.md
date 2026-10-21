# Dificuldade dos quizzes

`POST /api/quiz` e `POST /demo/quiz` aceitam o campo opcional `difficulty`: `BASIC`, `INTERMEDIATE` ou `ADVANCED`. Omitir o campo ou enviar `null` mantém o prompt anterior. Valores diferentes são recusados antes de chamar o modelo.

- Básico: compreensão de informações explícitas.
- Intermediário: aplicação em situações de trabalho simples.
- Avançado: análise e combinação de informações do material.

A dificuldade orienta o modelo; não é uma medida validada de complexidade pedagógica. O instrutor continua responsável por revisar perguntas, alternativas e explicações. Os níveis não permitem recorrer a conhecimentos externos ou inventar regras. A orientação é preservada nas novas tentativas quando o JSON gerado é inválido.

A interface de demonstração oferece o seletor. O formato do resultado permanece igual; não há mudanças na publicação de avaliações. Para oferecer essa escolha no LMS, seu frontend e backend precisam encaminhar o campo. Os testes locais verificam contratos e prompts com dublês; a qualidade pedagógica exige avaliação com o modelo real.
