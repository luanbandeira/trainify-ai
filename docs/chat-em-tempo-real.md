# Controle do chat em tempo real

As rotas `/api/chat/stream` e `/demo/chat/stream` compartilham um limite de execuções por processo. `CHAT_MAX_STREAMS` define esse limite (padrão: 4). Uma nova solicitação acima dele recebe HTTP 429 com `Retry-After: 10`, antes de iniciar a geração.

Cada execução guarda no máximo 64 eventos aguardando envio. Um consumidor lento faz o produtor aguardar, evitando acumular eventos indefinidamente. Ao encerrar a conexão, o servidor sinaliza o cancelamento e interrompe a geração no próximo envio de texto. O iterador da LLM é fechado ao sair do grafo.

Uma chamada ao provedor que já esteja bloqueada continua dependendo do timeout do SDK. A vaga só é liberada quando a execução termina; desconectar e reconectar não permite ultrapassar o limite. Erros também liberam a vaga.

O limite é local ao processo e específico das rotas de streaming. Ele não substitui cotas por empresa, proteção no gateway ou limites distribuídos em uma implantação com várias instâncias. As rotas de chat sem streaming e de quiz permanecem com seu comportamento anterior.

Os testes simulam saturação, falha e desconexão sem acessar um provedor pago. A validação com o provedor e o proxy usados na hospedagem ainda depende do ambiente publicado.
