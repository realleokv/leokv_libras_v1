# LibrasVision v2

Aplicação local em Python que usa a câmera do navegador para **coletar e reconhecer sinais isolados de um vocabulário treinado**. A interface está em português e oferece prévia das mãos, coleta guiada, treinamento, avaliação por classe e construção de texto revisável.

**Estado atual:** a versão 2 corrige a arquitetura e a validação da versão anterior. Ela começa sem modelos v2 treinados: é preciso coletar novas gravações reais. O catálogo inicial tem as 26 letras e expressões, mas isso não significa que todas sejam reconhecidas. As descrições são experimentais e estão marcadas como pendentes de revisão linguística. Não é um tradutor de Libras nem substitui intérprete.

## O que mudou

- Features v2 com 173 valores: configuração das duas mãos, pulsos, distância relativa, escalas e referências de nariz, boca e ombros.
- Movimento medido por deslocamento entre frames e tempo real; sequências de cerca de 2 segundos reamostradas para 32 frames.
- Estado e rastreadores separados por conexão; até 4 clientes simultâneos em um único processo.
- Emissão por execução: manter uma letra parada não a repete a cada dois segundos. Há confirmação manual opcional.
- Coleta realizada no servidor, com código de pessoa e sessão, cancelamento em interrupções e verificação de mãos/corpo.
- Dados reais separados por pessoa em treino, calibração de pontuações e teste. Há classe de controle `__NONE__` para repouso e transições.
- Modelos antigos e métricas antigas não são carregados. Alterar dados desativa imediatamente o modelo afetado.
- Administração com token, sessão de uma hora, CSRF, origens restritas, limites de imagens e de requisições, escrita transacional e treino em segundo plano.
- Renderização de textos cadastrados sem `innerHTML`; sem biblioteca JavaScript obrigatória de CDN.
- Texto reconhecido separado da revisão manual; nenhum verbo é acrescentado por suposição.
- A câmera permanece visível durante o treinamento. Abas acessíveis, navegação por teclado e avisos de erro.

## Requisitos

- Python **3.11 ou 3.12**, 64 bits. O ambiente de verificação usou Python 3.12 em Linux.
- Câmera e navegador recente com JavaScript. Para a câmera: `localhost` no computador ou HTTPS confiável em outro dispositivo.
- Alguns gigabytes livres para dependências. CPU é suficiente; esta versão usa PyTorch em CPU para previsibilidade.
- Internet na instalação das dependências. Depois, a interface e os modelos do MediaPipe usados aqui são locais.

Não atualize o MediaPipe isoladamente: esta implementação usa a API `solutions` da versão fixada. Se usar Python 3.13 ou superior, crie um ambiente com uma das versões suportadas.

## Instalação no Windows

No PowerShell, dentro da pasta do repositório:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Não é necessário ativar o ambiente ou alterar a política de execução do PowerShell. Se tiver Python 3.11, substitua `-3.12` por `-3.11`.

## Instalação no Linux

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements.txt
```

O OpenCV pode precisar das bibliotecas do sistema `libGL` e `libglib`. Em ambiente sem permissão administrativa, use um computador com essas bibliotecas instaladas; não é necessário contornar restrições da instituição.

## Configurar e executar

Gere dois valores diferentes, cada um com pelo menos 32 caracteres. Não use valores deste README como senhas reais. O arquivo `.env.example` é documentação; **não é carregado automaticamente**.

PowerShell:

```powershell
$env:LIBRAS_ADMIN_TOKEN = (& .\.venv\Scripts\python.exe -c "import secrets; print(secrets.token_urlsafe(32))")
$env:LIBRAS_SECRET_KEY = (& .\.venv\Scripts\python.exe -c "import secrets; print(secrets.token_hex(32))")
# Exiba o token somente no seu terminal e guarde-o para entrar na interface.
$env:LIBRAS_ADMIN_TOKEN
.\.venv\Scripts\python.exe app.py
```

Linux:

```bash
export LIBRAS_ADMIN_TOKEN="$(.venv/bin/python -c 'import secrets; print(secrets.token_urlsafe(32))')"
export LIBRAS_SECRET_KEY="$(.venv/bin/python -c 'import secrets; print(secrets.token_hex(32))')"
printf '%s\n' "$LIBRAS_ADMIN_TOKEN"
.venv/bin/python app.py
```

Abra **http://127.0.0.1:5000**. Na área **Coleta e treinamento**, entre com o token administrativo. Sem `LIBRAS_ADMIN_TOKEN`, o site abre em modo de reconhecimento, mas cadastro, coleta, exclusão e treino ficam bloqueados. Sem `LIBRAS_SECRET_KEY`, uma chave temporária segura é criada e as sessões deixam de valer ao reiniciar.

O token não é salvo no navegador. A autorização expira após uma hora; a saída revoga a coleta também nas conexões abertas da mesma sessão.

## Acesso por celular

O servidor escuta apenas no computador por padrão. Para abrir na rede local, use uma rede autorizada, o IP do computador e HTTPS. Exemplo, substituindo o IP pelo endereço correto:

```powershell
.\.venv\Scripts\python.exe app.py --host 0.0.0.0 --https --cert-host 192.168.1.20
```

No celular, o endereço será `https://192.168.1.20:5000`. `0.0.0.0` é um endereço de escuta, não o endereço para digitar no navegador.

O comando gera uma chave exclusiva e um certificado autoassinado em `data/tls/`, com localhost e os hosts fornecidos. **Um certificado autoassinado não garante acesso à câmera:** o dispositivo precisa confiar no certificado e no endereço. Para um certificado já confiável:

```powershell
.\.venv\Scripts\python.exe app.py --host 0.0.0.0 --https --cert caminho\certificado.pem --key caminho\chave.pem
```

O modo local renova o certificado autoassinado a cada inicialização; para manter confiança estável use seu próprio certificado com `--cert` e `--key`. Wi-Fi com isolamento entre clientes, firewall e bloqueios da escola podem impedir acesso mesmo com o código correto.

O servidor incluído é destinado a desenvolvimento local. Para exposição pública, use implantação com TLS confiável, proxy configurado, autenticação operacional e limites adequados. **Não execute múltiplos workers com este gerenciador em memória.**

## Coletar e treinar

1. Inicie a câmera. Para palavras com referência corporal, enquadre rosto e ombros. Para sinais de duas mãos, mantenha as duas visíveis.
2. Entre em Administração. Escolha o sinal, um código de pessoa (`p01`) e de sessão (`s01`). Use o mesmo código para a mesma pessoa em todas as sessões.
3. Prepare a gravação. Há contagem regressiva de 3 segundos e botão de cancelamento. Uma gravação estática usa 6–30 frames espaçados; o padrão é 12. Um gesto dinâmico deve ser executado completamente em cerca de 2 segundos.
4. Faça pelo menos duas execuções de cada classe com cada uma de três pessoas diferentes. Inclua `Nenhum sinal / transições` **separadamente nos modos estático e dinâmico**.
5. O treino precisa de pelo menos dois sinais e da classe de controle por modelo, seis gravações por classe e três pessoas. Todas as classes precisam aparecer nos três grupos separados. Esse mínimo apenas permite executar o treinamento: um conjunto maior e mais diverso é necessário para estimar qualidade.
6. Escolha estático, movimento ou ambos. Acompanhe o trabalho em segundo plano. Se um modelo falhar, o resultado de cada um será informado separadamente.
7. Confira o catálogo: só há métricas e disponibilidade para classes do modelo atual. A letra J segue indisponível enquanto não houver novas gravações e treinamento que a incluam.

A gravação é cancelada quando faltam mãos obrigatórias, muda a lateralidade, some a referência corporal necessária ou existe intervalo maior que 0,45 segundo entre frames. Conexões lentas podem exigir uma máquina mais rápida ou redução do custo de captura; não se deve colar trechos separados em uma única sequência.

Os códigos de pessoa determinam a separação da avaliação. Inventar códigos diferentes para a mesma pessoa invalida a independência do teste. Nenhuma identidade pessoal é verificada automaticamente.

## Avaliação e interpretação das métricas

- Pessoas inteiras ficam em grupos distintos de treino, calibração e teste. O código não retreina usando o teste depois de medir o resultado.
- Calibração por temperatura é feita em um grupo separado. Pontuação alta não garante reconhecimento correto, especialmente com poucos dados.
- Os limites padrão são 80% para estáticos e 85% para dinâmicos, com margem mínima entre as duas melhores classes. Os controles são individuais por conexão.
- A tabela informa precisão, F1 e número de gravações de teste de cada classe. Acurácia global se refere a gravações isoladas, não a tradução de conversas.
- Retire as mãos por pelo menos 0,4 segundo ou volte à classe de controle antes de repetir o mesmo sinal. Também é possível adicionar manualmente o sinal atual.
- O catálogo inclui campo para fonte ou vídeo de referência e responsável pela revisão. Não há vídeos ou validação linguística fabricados nesta entrega.
- Não há reconhecimento de expressões faciais nem análise linguística completa. A pose fornece referências de posição, não interpretação de todas as marcas não manuais.

### Avaliar vídeos completos

O comando abaixo percorre detecção, extração, classificador e estabilizador. Ele mede acertos por evento, sinais perdidos, falsas emissões/repetições e tempo de processamento. Use vídeos reais de pessoas independentes, com autorização.

Copie `tools/video_manifest.example.json`, ajuste os caminhos e anote começo/fim de cada sinal. Os caminhos dos vídeos são relativos ao manifesto. Vídeos sem sinais usam `"events": []`.

```powershell
.\.venv\Scripts\python.exe tools/evaluate_video.py meu_manifesto.json --output reports/avaliacao.json
```

A tolerância para uma emissão é de até 0,5 segundo depois do fim anotado. Este comando não mede qualidade da tradução linguística. Nenhum percentual de webcam real é prometido sem executar esse protocolo em gravações apropriadas.

## Dados e migração

- A aplicação cria `data/libras_v2.db` e `data/models_v2/`. A pasta é ignorada pelo Git.
- Cada gravação guarda apenas features, timestamps, pessoa, sessão, origem `real` e versão 2. Não salva JPEGs nem vídeos da câmera.
- SQLite usa transações e WAL. Treino e alterações administrativas são serializados. Modelos são publicados em pastas versionadas com troca atômica do registro.
- Adicionar ou apagar amostras desativa o modelo daquele tipo até um novo treino. Métricas antigas deixam de aparecer como atuais.
- `legacy/` preserva os datasets, modelos e banco da versão anterior. Eles não são importados nem executados; o banco legado inclui histórico anteriormente versionado. Não os use como validação independente.
- Os dados antigos de 146 features não contêm as informações novas; não é possível recuperar trajetória/corpo que nunca foram gravados. Faça nova coleta, incluindo J e negativos.
- O gerador `complete_database.py` foi desativado como caminho de treinamento e apenas explica a nova coleta. Não fabrica mãos, trajetórias ou precisão.
- `key.pem` e `cert.pem` foram removidos. A chave antiga esteve pública: não a reutilize. A remoção atual não apaga commits históricos; eventual reescrita do histórico deve ser coordenada com os colaboradores.
- Para backup, pare o servidor e copie `data/` inteira. O registro e as versões antigas de modelos podem ser preservados para recuperação; só o modelo apontado no registro é carregado.

## Segurança e privacidade

O acesso normal ao site é público para quem alcança o servidor. Cadastro, gravação, treino, exclusão e histórico exigem administração. Não compartilhe token administrativo e chave de sessão. Apenas configure `LIBRAS_ALLOWED_ORIGINS` se necessário, informando origens completas separadas por vírgula; o padrão exige a mesma origem. Não há curinga.

Imagens JPEG têm limite de 400 KB, no máximo 1280 pixels por dimensão e 1.228.800 pixels no total. O servidor limita conexões e frequência de frames, tentativas de login e coletas. Mensagens internas de exceção ficam no terminal.

O histórico no banco fica **desativado por padrão**. `LIBRAS_LOG_HISTORY=1` habilita no máximo 1.000 eventos, com data/hora UTC completa e identificador de conexão. O administrador pode consultar `/api/history` e apagar pelo botão da interface. A lista exibida de últimos sinais fica na memória do navegador.

Modelos estáticos usam `skops`; o PyTorch carrega apenas pesos. Arquivos legados com pickle não são abertos. Hashes detectam corrupção; não substituem autenticidade. Use somente modelos criados nesta instalação e mantenha a pasta `data/` protegida pelo sistema operacional.

## Estrutura de pastas

| Caminho | Função |
|---|---|
| `app.py` | Rotas, sessão administrativa, WebSocket, limites e HTTPS local |
| `config.py` | Parâmetros e catálogo inicial experimental |
| `database.py` | Catálogo, gravações, revisões, métricas e histórico |
| `feature_extractor.py` | MediaPipe Hands e Pose, geometria e velocidade |
| `temporal.py` | Validação temporal e reamostragem |
| `capture.py` | Coleta guiada e validação no servidor |
| `ml_engine.py` | Random Forest, LSTM, separação por pessoa e calibração |
| `stabilizer.py` | Estabilização, rejeição e controle de repetição |
| `templates/index.html` | Interface acessível em português |
| `static/js/main.js` | Câmera, comunicação, coleta e administração |
| `static/js/text_builder.js` | Construção local de texto com tokens tipados |
| `static/css/style.css` | Layout, responsividade e acessibilidade |
| `static/vendor/` | Socket.IO local e sua licença |
| `tests/` | Regressões de Python e JavaScript |
| `tools/evaluate_video.py` | Avaliação ponta a ponta com vídeos anotados |
| `legacy/` | Arquivos antigos, não usados pela aplicação |
| `data/` | Dados privados criados localmente, fora do Git |

## Testes

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
node --test tests/text_builder.test.cjs
```

No Linux, use `.venv/bin/python`. Node.js é necessário apenas para os testes JavaScript, não para executar o site.

Os testes usam coordenadas artificiais e rastreadores simulados para verificar regras, segurança e persistência; **não são evidência de precisão real em Libras**. Os testes de treinamento exercitam gravação, partições e carregamento dos dois modelos. Validação linguística e testes reais de câmera continuam exigindo pessoas e gravações adequadas.

## Solução de problemas

- **Sem modelo disponível:** é o estado esperado antes da nova coleta. Não copie modelos de `legacy/` para a pasta v2.
- **Coleta cancelada:** mantenha as mãos/corpo visíveis, confira o sinal selecionado e a estabilidade da conexão.
- **Treino recusado:** leia a classe indicada; confira pessoas reais distintas, número de gravações e a classe de controle em cada modo.
- **Câmera bloqueada:** confira permissão, aplicativo concorrente e contexto seguro. Um endereço HTTP de rede não equivale a localhost.
- **MediaPipe indisponível:** confira Python 3.11/3.12 e reinstale `requirements.txt` em ambiente novo. Detalhes ficam no terminal e `/api/health` mostra o estado resumido.
- **Servidor desconectado:** o cliente tenta reconectar; coletas interrompidas devem ser repetidas. Depois de reiniciar o servidor com chave temporária, recarregue a página.
- **Modelo desativado após coleta/exclusão:** comportamento intencional; treine novamente para reconciliar classes, dados e métricas.
- **A letra mantida não se repete:** retire as mãos brevemente e execute de novo, ou confirme manualmente.

## Limites que exigem trabalho de campo

Para demonstrar reconhecimento confiável, ainda é necessário coletar dados reais diversos, revisar o catálogo com profissional de Libras, adicionar demonstrações autorizadas e medir vídeos independentes. O código agora oferece mecanismos para esse trabalho e evita apresentar o resultado antigo como se essa validação já tivesse sido feita.
