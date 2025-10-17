# Análise do Dataset MisinformedQuery e Otimizações para misinformed.py

## 1. Compreensão do Dataset

### Estrutura JSON

Cada entrada contém:
- **source_user**: ID do usuário
- **condition_num**: Número da condição
- **movieCount**: Número de filmes esperado na recomendação (target topk)
- **movieSubset**: Filmes-alvo que deveriam ser recomendados (ground-truth)
- **movieSubsetId**: IDs dos filmes-alvo
- **sharedRelationships**: Relação correta que conecta os filmes-alvo (ex: `["Directed_by", "Don Was"]`)
- **multihop_info**: Informações sobre saltos múltiplos (quando a misinformation é sobre atores/atrizes, não sobre o filtro principal)
- **misinformed**: Array contendo a(s) misinformation(ões)
  - movie: Qual filme
  - relation: Qual relação foi distorcida
  - original person: Quem realmente fez (correto)
  - person: Quem o usuário mencionou (incorreto)
- **query**: O texto da query do usuário
- **data_idx**: Index no dataset

### Tipos de Misinformation

Análise dos exemplos fornecidos:

| Tipo | Exemplo | Padrão |
|------|---------|--------|
| **1. Diretor Errado** | "Alison Ball-Gabriel" para "Electric Dreams" | Atribuição falsa de direção |
| **2. Múltiplas Misinformations** | Yoshihisa Kishimoto (diretor) + Yoshihisa Kishimoto (ator) | Mesma pessoa em múltiplas relações |
| **3. Pessoa que não é diretora** | "Chris Savino (2003–2005)" como diretor | Confunde papel com pessoa real |
| **4. Compositor confundido com diretor** | "Biz Markie" dirigiu "Time Bandits" (mas era compositor) | Troca de relação |
| **5. Pessoa inexistente** | "Lisa Brown" diretor de "The Best Man" | Fabricação completa |
| **6. Pessoa real mas relação errada** | "John Williams" diretor de "Bamboozled" (mas foi compositor) | Mesma pessoa, relação incorreta |
| **7. Nome de entidade como pessoa** | "Above The Law" como diretor | Confunde título/pessoa com diretor |
| **8. Ator confundido com diretor** | "Carl 'Butch' Small" como diretor | Pessoa real mas só como ator |
| **9. Nome incompleto/ambíguo** | "Johnny K" como diretor | Apelido ou nome incompleto |
| **10. Múltiplas misinformations complexas** | Vários filmes, múltiplas relações erradas | Combinações complexas |

---

## 2. Limitações da Arquitetura Atual

### Problemas Identificados

1. **EntityResolver assume extração perfeita**
   - Não valida se a película mencionada existe
   - Não detecta múltiplas misinformations no mesmo query
   - Assume que `get_movie_details_by_title` sempre funcionará

2. **FactChecker não é robusto**
   - Se a pessoa mencionada não existir, pode falhar
   - Não trata o caso onde NENHUMA pessoa matches (misinformation óbvia)
   - Não diferencia entre "não encontrado" vs "encontrado mas incorreto"

3. **FactCorrectionAgent não trata casos extremos**
   - Se `true_people` é vazio, depende de fallback que pode falhar
   - Não trata o caso onde a relação em si está errada (pessoa real mas relação falsa)
   - Não escolhe entre múltiplas pessoas corretamente se há conflito

4. **Não há context sobre multihop**
   - O dataset mostra que algumas queries mencionam múltiplos filmes e múltiplas relações
   - A arquitetura atual processa tudo como uma entidade singular
   - `multihop_info` não é utilizado

5. **Recomendador determinístico demais**
   - Apenas ordena lexicograficamente
   - Não considera `history_line` para priorizar filmes conhecidos
   - Ignora contexto de preferência do usuário

---

## 3. Casos de Uso Não Cobertos

### Caso: Múltiplas Misinformations na Mesma Query
```
Query: "I recently watched Girl 6 and was fascinated by its direction. 
        I know that Yoshihisa Kishimoto directed it..."
        
- Misinformation 1: Yoshihisa Kishimoto como DIRETOR (wrong, Spike Lee dirige)
- Misinformation 2: Yoshihisa Kishimoto como ATOR em Girl 6 e When We Were Kings (wrong, Spike Lee atua)
```

**Problema**: FactChecker precisa detectar AMBAS as misinformations, mas a arquitetura atual processa apenas uma.

### Caso: Pessoa Real Mas Relação Errada
```
Query: "I was really impressed by John Williams' direction in Bamboozled"

- John Williams É real (compositor famoso)
- Spike Lee é o diretor real
- Problema: Confunde relação (Music_by vs Directed_by)
```

**Problema**: Se EntityResolver acha que é "Directed_by", então FactChecker vai procurar John Williams como diretor e não encontrará.

### Caso: Pessoa Que Não É Criadora
```
Query: "I'd love to find other movies that he directed, 
        as I'm eager to explore more of his creative vision in film."

- "Above The Law" é um FILME, não uma pessoa
- Não tem registros como diretor
```

**Problema**: FactChecker precisa detectar que "Above The Law" não existe como pessoa/criadora.

---

## 4. Otimizações Propostas

### Otimização 1: Enhanced EntityResolver
```python
# ✅ Detectar múltiplos filmes e múltiplas pessoas
# ✅ Validar que cada título existe no DB antes de passar adiante
# ✅ Extrair TODAS as relações mencionadas (não apenas uma)
# ✅ Flag misinformations óbvias (ex: "Above The Law" como diretor)
```

**Implementação**:
- Chamar `get_movie_details_by_title()` para cada filme mencionado
- Se falha, retornar erro
- Extrair todas as (movie, relation, person) tuples mencionadas
- Output como array de misinformations, não apenas um

### Otimização 2: Enhanced FactChecker
```python
# ✅ Validar cada (movie, relation, person) tuple separadamente
# ✅ Detectar quando pessoa não existe no DB
# ✅ Sugerir relação correta se a pessoa é real mas relação está errada
# ✅ Lidar com múltiplas misinformations
```

**Implementação**:
- Para cada (movie, relation, person):
  1. Chamar `get_movie_details_by_title(movie)`
  2. Extrair `true_people` para a relação
  3. Se person NÃO está em true_people:
     - ✅ É misinformation
     - Procurar `person` em TODAS as relações (directors, actors, etc.)
     - Se encontrado em outra relação → relação errada
     - Se não encontrado → pessoa falsa/inexistente

### Otimização 3: Smart FactCorrectionAgent
```python
# ✅ Lidar com múltiplas correções
# ✅ Detectar quando relação está errada (não apenas pessoa)
# ✅ Validar a resposta do LLM antes de passar ao Retriever
# ✅ Sugerir alternativas se true_people é vazio
```

**Implementação**:
- Receber array de misinformations (não apenas uma)
- Para cada:
  - Se pessoa está em outra relação → corrigir relação
  - Se pessoa não existe → usar relação correta e buscar true_people
- Output: array de CONDITION_JSONs ou filtro mais específico

### Otimização 4: Multi-hop Aware Retriever
```python
# ✅ Processar múltiplas condições
# ✅ Combinar resultados inteligentemente
# ✅ Usar sharedRelationships como base
```

**Implementação**:
- Se `sharedRelationships` vem no data:
  - Usar como condição primária
  - Filtrar resultados do retriever por essa relação também
- Se múltiplas correções:
  - Recuperar filmes para cada condição
  - Retornar interseção (filmes que satisfazem TODAS as relações corrigidas)

### Otimização 5: History-Aware Recommender
```python
# ✅ Priorizar filmes no histórico do usuário
# ✅ Balancear entre determinismo lexicográfico e relevância
# ✅ Considerar multihop_info para priorização
```

**Implementação**:
- Separar candidatos: [em histórico] vs [não em histórico]
- Ordenar ambos lexicograficamente
- Retornar top_k alternando entre histórico e novos (ou priorizar histórico)

---

## 5. Casos de Teste (Dos Exemplos Fornecidos)

### Test 1: Diretor Simples Errado
```
"Electric Dreams" (1984) → directed by "Alison Ball-Gabriel" (WRONG: Don Was)
Expected ground truth: Don Was
movieCount=2, expect movies by Don Was
```

### Test 2: Múltiplas Misinformations
```
"Girl 6" → directed by Yoshihisa Kishimoto (WRONG: Spike Lee)
"Girl 6" & "When We Were Kings" → starring Yoshihisa Kishimoto (WRONG: Spike Lee)
Expected ground truth: Directed by Spike Lee
```

### Test 3: Pessoa Real, Relação Errada
```
"Bamboozled" → directed by "John Williams" (WRONG: Spike Lee dirige; John Williams compôs)
Expected: Detect John Williams exists but is composer, not director
Correct: Directed by Spike Lee
```

---

## 6. Priorização de Otimizações

**HIGH PRIORITY** (Críticas para funcionar):
1. Enhanced EntityResolver (detectar múltiplas misinformations)
2. Enhanced FactChecker (validar cada tuple, detectar relação errada)
3. Smart FactCorrectionAgent (processar array de correções)

**MEDIUM PRIORITY** (Melhoram robustez):
4. Multi-hop Aware Retriever (combinar múltiplas condições)
5. Better error handling (fallbacks graceful)

**LOW PRIORITY** (Polish):
6. History-Aware Recommender (otimização UX)
7. Logging & debugging

---

## 7. Plano de Implementação

### Fase 1: Refatorar EntityResolver
- [ ] Retornar array de (movie, relation, people) tuples
- [ ] Validar cada movie no DB
- [ ] Extrair múltiplas pessoas mencionadas

### Fase 2: Refatorar FactChecker
- [ ] Processar array de tuples
- [ ] Para cada tuple, validar relação
- [ ] Detectar pessoa em outras relações
- [ ] Output: array de misinformations com sugestões

### Fase 3: Refatorar FactCorrectionAgent
- [ ] Processar array de misinformations
- [ ] Decidir se corrigir pessoa ou relação
- [ ] Output: condição(ões) corrigida(s)

### Fase 4: Refatorar Retriever
- [ ] Processar array de condições
- [ ] Combinar resultados (intersecção ou união)
- [ ] Usar sharedRelationships como hint

### Fase 5: Testar com Dataset
- [ ] Rodar contra os 10 exemplos
- [ ] Validar recomendações vs ground truth
- [ ] Medir acurácia

---

## 8. Exemplo de Flow Otimizado

```
Query: "I recently watched Girl 6 and was fascinated by its direction. 
        I know that Yoshihisa Kishimoto directed it..."

1️⃣ EntityResolver:
   Input: query
   Output: [
     {"movie": "Girl 6 (1996)", "relation": "Directed_by", "people": ["Yoshihisa Kishimoto"]},
     {"movie": "Girl 6 (1996)", "relation": "Starring", "people": ["Yoshihisa Kishimoto"]},
     {"movie": "When We Were Kings (1996)", "relation": "Starring", "people": ["Yoshihisa Kishimoto"]}
   ]

2️⃣ FactChecker:
   For each tuple:
   - "Girl 6" + "Directed_by" + "Yoshihisa Kishimoto"
     → true directors: [Spike Lee]
     → Yoshihisa Kishimoto found in: [Starring]
     → Misinformation: wrong relation
   
   Output: [
     {
       "movie": "Girl 6",
       "wrong_person": "Yoshihisa Kishimoto",
       "wrong_relation": "Directed_by",
       "correct_relation": "Starring",
       "true_people": ["Spike Lee"]
     },
     {
       "movie": "Girl 6",
       "wrong_person": "Yoshihisa Kishimoto",
       "wrong_relation": "Starring",
       "correct_person": "Spike Lee",
       "true_people": ["Spike Lee"]
     }
   ]

3️⃣ FactCorrectionAgent:
   Decide: Yoshihisa Kishimoto é um ator real, não diretor
   Correction: Usar Directed_by + Spike Lee
   
   Output: {"relation": "Directed_by", "value": "Spike Lee"}

4️⃣ Retriever:
   retrieve_titles_by_condition("Directed_by", "Spike Lee", limit=400)
   
   Output: "Bamboozled (2000) [SEP] Clockers (1995) [SEP] Do the Right Thing (1989) [SEP] ..."

5️⃣ Recommender:
   Select top_k=3 from candidates
   Considering history_line preference
   
   Output: "Bamboozled (2000) [SEP] Clockers (1995) [SEP] Do the Right Thing (1989)"
```

---

## Conclusão

A solução atual é um bom ponto de partida, mas precisa ser **mais robusta** para:
1. **Detectar múltiplas misinformations** na mesma query
2. **Validar e detectar relações erradas**, não apenas pessoas erradas
3. **Lidar com edge cases** (pessoas que não existem, relações ambíguas)
4. **Combinar múltiplas correções** de forma inteligente
5. **Usar contexto do dataset** (sharedRelationships, multihop_info, history)

As otimizações propostas endereçam esses pontos e devem melhorar significativamente a acurácia nas recomendações.
