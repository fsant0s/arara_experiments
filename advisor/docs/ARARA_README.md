# README — Guia de uso do ARARA (baseado nos notebooks de `examples/`)

## 1) visão geral

O ARARA, pelos exemplos da pasta `examples/`, organiza aplicações em torno de:

- **`User`** (origem da mensagem)
- **`Agent`** (execução de tarefas, tools e skills)
- **`Module`** (agrupamento de agentes + regras de fluxo)
- **`Orchestrator`** (coordena o fluxo entre agentes/módulos)

O início do fluxo acontece com `user.talk_to(...)`, direto com um agente ou passando por um orquestrador.

---

---

## 3) mapa do fluxo (user -> orquestrador -> agentes -> resposta)

Fluxo padrão observado:

1. `User` envia a mensagem inicial (`talk_to`)
2. `Orchestrator` seleciona o próximo speaker com base no `Module`
3. Agentes executam suas funções (LLM, tools, skills, memória)
4. Resposta volta ao usuário

Exemplo de início:
- `user.talk_to(arara_orc, message="Me recomende um filme")` em `examples/modules.ipynb`

---

## 4) orquestradores

Uso observado em `examples/modules.ipynb` e `examples/hiaac.ipynb`:

- Criação com `Orchestrator(name, module, llm_config, system_message?, description?)`
- Recebe um `Module` com os agentes e regras de transição
- Pode existir em camadas (orquestrador principal + orquestrador especializado)
- O controle de fluxo depende de:
  - `speaker_selection_method` (`auto`, `round_robin`)
  - `allowed_or_disallowed_speaker_transitions`
  - `speaker_transitions_type="allowed"`

---

## 5) agentes e comunicação

Criação de agentes:

- `Agent(name, llm_config, system_message?, description?, tools?, skills?, ...)`

Padrões de comunicação:

- `user.talk_to(agent, message="...")` (direto)
- `user.talk_to(orchestrator, message="...")` (roteado)

### Exemplo de `send()`

```python
# Envia mensagem para outro agente sem esperar resposta imediata
self.send(
    message="Atualizei o contexto com as preferências do usuário.",
    recipient=agent,      # agente destino
    request_reply=False,  # não aguarda resposta síncrona
    silent=True,          # envio silencioso
)
```

Parâmetros do `send()`:

- `message`: conteúdo enviado
- `recipient` (ou `agent`): destinatário
- `request_reply`:
  - `True`: pede resposta
  - `False`: envio sem aguardar retorno imediato
- `silent`:
  - `True`: sem saída/log padrão
  - `False`: com saída normal

---

## 6) memórias e tipos de memória

### memória episódica (`EpisodicMemory`)
Referência: `examples/episodic_memory.ipynb`

- Inicialização:
  - `episodic_memory = EpisodicMemory(name="chat_history")`
  - `episodic_memory.add(MemoryContent(content="User is vegan."))`
- Acoplamento:
  - `episodic_skill = EpisodicMemorySkill(episodic_memory=episodic_memory)`
  - `Agent(..., skills=[episodic_skill])`
- Efeito prático: o agente usa o histórico para escolher ação/tool adequada.

### memória compartilhada (`SharedMemory`)
Referência: `examples/shared_memory.ipynb`

- Inicialização:
  - `shared_memory = SharedMemory(name="shared_chat_history")`
  - `shared_skill = SharedMemorySkill(shared_memory=shared_memory)`
- Dois agentes compartilham a mesma memória.
- Helpers observados no agente:
  - `add_shared_memory(...)`
  - `list_shared_memory()`
  - `list_shared_memory_by_owner("agent_b")`
  - `update_shared_memory(id, ...)`
  - `remove_shared_memory(id)`

---

## 7) módulos

Referências: `examples/modules.ipynb`, `examples/hiaac.ipynb`

`Module` define o “espaço de conversa” e as regras de transição:

- `name`
- `agents=[...]`
- `speaker_selection_method="auto" | "round_robin"`
- `allowed_or_disallowed_speaker_transitions={...}`
- `speaker_transitions_type="allowed"`

Pode ser usado para:
- pipeline especializado interno
- módulo principal de interação com usuário

---

## 8) execução da arquitetura (passo a passo)

1. Defina `llm_config`
2. Instancie o `User`
3. Crie os `Agent`s (com `tools`/`skills` conforme necessidade)
4. Configure memória (episódica ou compartilhada), se aplicável
5. Monte `Module` com regras de transição
6. Crie `Orchestrator` com o módulo
7. Inicie com mensagem do usuário:
   - `user.talk_to(orchestrator, message="...")`
8. Leia a resposta final e, se necessário, itere a conversa

---

## 9) funções/métodos principais

- `User.talk_to(target, message=..., cache=...)`
- `Agent(...)`
- `Module(...)`
- `Orchestrator(...)`
- `self.send(message, recipient, request_reply=..., silent=...)`
- `EpisodicMemory(...)`
- `SharedMemory(...)`
- `MemoryContent(...)`
- `EpisodicMemorySkill(...)`
- `SharedMemorySkill(...)`
- `add_shared_memory(...)`
- `list_shared_memory()`
- `list_shared_memory_by_owner(...)`
- `update_shared_memory(...)`
- `remove_shared_memory(...)`

Skills mostradas nos notebooks:

- `WebCrawler` (`examples/web_crawler.ipynb`, `examples/hiaac.ipynb`)
- `WebSearch` (`examples/web_search.ipynb`)
- `GoogleDocsReader` (`examples/google_docs_reader.ipynb`, `examples/hiaac.ipynb`)

Code execution:

- `LocalCommandLineCodeExecutor` + `code_execution_config={"executor": executor}` em `examples/code_executor.ipynb`

---

## 10) boas práticas observadas

- Definir `description` e `system_message` claros por agente
- Separar responsabilidades por agentes especializados
- Restringir transições com dicionário explícito de speakers
- Anexar memória como skill
- Usar tools explícitas para lógica determinística
- Centralizar fluxo em `Module` + `Orchestrator` quando há múltiplos agentes

---

## 11) limitações da análise

- Escopo restrito aos notebooks de `examples/`
- Não cobre variações de API não demonstradas nesses notebooks
- Não cobre detalhes internos completos de implementação fora dos exemplos

---

## 12) cheat sheet

- Criar usuário:
  - `user = User(name="user")`
- Criar agente:
  - `agent = Agent(name="x", llm_config=..., tools=[...], skills=[...])`
- Criar módulo:
  - `Module(..., agents=[...], speaker_selection_method="auto", allowed_or_disallowed_speaker_transitions=..., speaker_transitions_type="allowed")`
- Criar orquestrador:
  - `orc = Orchestrator(name="...", module=module, llm_config=...)`
- Iniciar fluxo:
  - `user.talk_to(orc, message="...")`
- Enviar entre agentes:
  - `self.send(message="...", recipient=agente, request_reply=False, silent=True)`
- Memória episódica:
  - `EpisodicMemory` + `EpisodicMemorySkill`
- Memória compartilhada:
  - `SharedMemory` + `SharedMemorySkill` + helpers de CRUD/listagem

---

## 13) template mínimo de execução (pseudocódigo consolidado)

```python
from agents import User, Agent, Module, Orchestrator
from capabilities.memories import EpisodicMemory, SharedMemory, MemoryContent
from capabilities.skills import EpisodicMemorySkill, SharedMemorySkill

# 1) Configuração do LLM
llm_config = {...}

# 2) Usuário
user = User(name="user")

# 3) Memórias (opcional)
episodic = EpisodicMemory(name="chat_history")
episodic.add(MemoryContent(content="Usuário prefere conteúdo objetivo"))
episodic_skill = EpisodicMemorySkill(episodic_memory=episodic)

shared = SharedMemory(name="shared_chat_history")
shared_skill = SharedMemorySkill(shared_memory=shared)

# 4) Agentes
agent_a = Agent(name="agent_a", llm_config=llm_config, skills=[shared_skill])
assistant = Agent(
    name="assistant",
    llm_config=llm_config,
    skills=[episodic_skill],
    tools=[...],
)

# 5) Exemplo de comunicação direta entre agentes
assistant.send(
    message="Contexto atualizado",
    recipient=agent_a,
    request_reply=False,
    silent=True,
)

# 6) Regras de transição e módulo
transitions = {
    user: [assistant],
    assistant: [user],
}
module = Module(
    name="main_module",
    agents=[user, assistant],
    speaker_selection_method="auto",
    allowed_or_disallowed_speaker_transitions=transitions,
    speaker_transitions_type="allowed",
)

# 7) Orquestrador
orc = Orchestrator(
    name="main_orchestrator",
    module=module,
    llm_config=llm_config,
)

# 8) Execução ponta a ponta
result = user.talk_to(orc, message="Quero uma recomendação")
print(result)
```
