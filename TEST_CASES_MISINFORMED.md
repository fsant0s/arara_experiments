# Test Cases: Misinformed Queries v1 vs v2

## Overview
Este documento contém 10 test cases baseados no dataset MisinformedQuery.json, 
com expected outputs para v1 vs v2.

---

## Test Case 1: Diretor Simples Errado

### Input Data
```json
{
  "source_user": "1354",
  "query": "I just watched Electric Dreams and was fascinated by the direction of Alison Ball-Gabriel. I'm eager to find other movies directed by her to see her unique style again.",
  "movieCount": 2,
  "movieSubset": ["Anywhere But Here (1999)", "Hope Floats (1998)"],
  "sharedRelationships": [["Directed_by", "Don Was"]]
}
```

### Expected Ground Truth
- **Correct Relation**: Directed_by
- **Correct Person**: Don Was
- **Movies**: Anywhere But Here (1999), Hope Floats (1998)

### v1 Behavior
```
EntityResolver: 
  - titles: ["Electric Dreams (1984)"]
  - people: ["Alison Ball-Gabriel"]
  - relation_hint: "Directed_by"

FactChecker:
  - Calls get_movie_details_by_title("Electric Dreams (1984)")
  - Directors: [Don Was]
  - Alison Ball-Gabriel NOT found
  - ✅ Output: wrong_person, true_people=[Don Was]

FactCorrectionAgent:
  - ✅ Selects: Don Was

Retriever:
  - ✅ retrieve_titles_by_condition("Directed_by", "Don Was")
  - Result: [...movies by Don Was...]

Recommender:
  - ✅ Top-2 by alpha order
  - Expected: "Anywhere But Here (1999) [SEP] Hope Floats (1998)"

Accuracy: ✅ PASS
```

### v2 Behavior
```
EntityResolver:
  - ✅ Same (single misinformation)
  - Validates "Electric Dreams (1984)" exists

FactChecker:
  - ✅ Same logic but structured as array analysis
  - Confirms: Alison Ball-Gabriel not found anywhere
  - reason: "person_not_found"

FactCorrectionAgent:
  - ✅ Confirms: primary_condition = (Directed_by, Don Was)

Retriever:
  - ✅ Same

Recommender:
  - ✅ Same (or better with history awareness)

ResponseGenerator:
  - ✅ Generates: "Actually, Electric Dreams (1984) was directed by Don Was, not Alison Ball-Gabriel..."

Accuracy: ✅ PASS (+ educational response)
```

---

## Test Case 2: Múltiplas Misinformations (Mesmo Filme, Múltiplas Relações)

### Input Data
```json
{
  "source_user": "4647",
  "query": "I recently watched Girl 6 and was fascinated by its direction. I know that Yoshihisa Kishimoto directed it, and I'm eager to find other films that he has directed. Additionally, I noticed that he starred in both Girl 6 and When We Were Kings, so if any of his directed films also feature him, that would be great!",
  "movieCount": 3,
  "movieSubset": ["Do the Right Thing (1989)", "Bamboozled (2000)", "Clockers (1995)"],
  "sharedRelationships": [["Directed_by", "Spike Lee"]]
}
```

### Expected Ground Truth
- **Correct Relation**: Directed_by
- **Correct Person**: Spike Lee
- **Misinformation 1**: Yoshihisa Kishimoto directed Girl 6 (WRONG: Spike Lee directed)
- **Misinformation 2**: Yoshihisa Kishimoto starred in Girl 6 and When We Were Kings (WRONG: Spike Lee starred)

### v1 Behavior
```
EntityResolver:
  - titles: ["Girl 6 (1996)"]
  - people: ["Yoshihisa Kishimoto"]
  - relation_hint: "Directed_by" (picks first mentioned)
  ❌ MISSES the Starring mentions!

FactChecker:
  - Only checks: Yoshihisa Kishimoto as Directed_by
  - ❌ Never checks him as actor
  - ❌ Result: Incorrect or incomplete analysis

Accuracy: ❌ FAIL (missed multiple misinformations)
```

### v2 Behavior
```
EntityResolver:
  - Extrai AMBAS as menções:
    1. {"movie": "Girl 6 (1996)", "relation": "Directed_by", "mentioned_person": "Yoshihisa Kishimoto"}
    2. {"movie": "Girl 6 (1996)", "relation": "Starring", "mentioned_person": "Yoshihisa Kishimoto"}
    3. {"movie": "When We Were Kings (1996)", "relation": "Starring", "mentioned_person": "Yoshihisa Kishimoto"}
  - ✅ Validates all movies exist

FactChecker:
  - For each tuple:
    1. Girl 6 + Directed_by + Yoshihisa: NOT found in directors, FOUND in actors
       → reason: "person_found_in_other_relation"
       → actual_relation: "Starring"
       → correct_person: "Spike Lee"
    2. Girl 6 + Starring + Yoshihisa: NOT found in actors
       → reason: "person_not_found"
       → correct_person: "Spike Lee"
    3. When We Were Kings + Starring + Yoshihisa: NOT found in actors
       → reason: "person_not_found"
       → correct_person: "Spike Lee"

FactCorrectionAgent:
  - Reconhece: Yoshihisa não dirigiu, Spike Lee dirigiu
  - Primary condition: (Directed_by, Spike Lee)

Retriever:
  - ✅ Finds Spike Lee films

Recommender:
  - ✅ Top-3

Accuracy: ✅ PASS (detected all misinformations)
```

---

## Test Case 3: Pessoa Real, Relação Errada

### Input Data
```json
{
  "source_user": "839",
  "query": "I recently watched Bamboozled and was really impressed by John Williams' direction. I'd love to find more movies that he directed. If John Williams happens to star in any of them, that would be a great bonus!",
  "movieCount": 2,
  "movieSubset": ["Do the Right Thing (1989)", "Clockers (1995)"],
  "sharedRelationships": [["Directed_by", "Spike Lee"]]
}
```

### Expected Ground Truth
- **Correct Relation**: Directed_by (Spike Lee)
- **Misinformation**: John Williams directed Bamboozled (WRONG: John Williams composed, Spike Lee directed)

### v1 Behavior
```
FactChecker:
  - Calls get_movie_details_by_title("Bamboozled (2000)")
  - Checks: directors = [Spike Lee]
  - John Williams NOT in directors
  - ❓ Output: wrong_person (but doesn't explain WHAT John Williams is)
  - LLM might assume he's fake, not that he's a composer

FactCorrectionAgent:
  - Selects Spike Lee
  - ❌ Doesn't explain that John Williams is a real person (composer)
  - ❌ User confusion: "But John Williams IS a famous person!"

Accuracy: ⚠️ PARTIAL (correct rec, but wrong diagnosis)
```

### v2 Behavior
```
FactChecker:
  - John Williams NOT found in directors
  - ✅ SEARCHES in ALL other relations
  - Found in composers (Music_by)!
  - reason: "person_found_in_other_relation"
  - actual_relation: "Music_by"

FactCorrectionAgent:
  - Reconhece: John Williams is a real person (composer)
  - Corrected relation: Directed_by
  - Corrected person: Spike Lee
  - Explanation: "John Williams composed the music, but Spike Lee directed the film"

ResponseGenerator:
  - "Bamboozled was directed by Spike Lee, not John Williams. 
    John Williams (who composed the score) is a different role. 
    Here are other films directed by Spike Lee..."

Accuracy: ✅ PASS (+ educational + correct diagnosis)
```

---

## Test Case 4: Pessoa Totalmente Inexistente

### Input Data
```json
{
  "source_user": "202",
  "query": "I just watched The Best Man (1999) and was really impressed by Lisa Brown's direction. I'm looking for other films that she has directed. If you could recommend some, that would be great!",
  "movieCount": 2,
  "movieSubset": ["..."],
  "sharedRelationships": [...]
}
```

### Expected Ground Truth
- Lisa Brown did NOT direct The Best Man (1999)
- Need to find correct director and recommend their films

### v1 Behavior
```
FactChecker:
  - Calls get_movie_details_by_title("The Best Man (1999)")
  - Lisa Brown NOT found in directors
  - ✅ Detects wrong_person
  - ❌ BUT: true_people might be empty or cause LLM confusion
  - ❌ Doesn't check other relations (Lisa Brown might be actress)

Accuracy: ⚠️ PARTIAL (might fail on fallback logic)
```

### v2 Behavior
```
FactChecker:
  - Lisa Brown NOT found in directors
  - ✅ Searches in ALL relations
  - Checks: actors, composers, producers, writers
  - ❓ If found in ANY: reason="person_found_in_other_relation"
  - If NOT found: reason="person_not_found"
  - Finds true directors from The Best Man (1999)

FactCorrectionAgent:
  - Fallback logic: Use true directors of The Best Man
  - Primary condition: (Directed_by, true_director)

Retriever:
  - ✅ Retrieves films by true director

Accuracy: ✅ PASS (graceful fallback)
```

---

## Test Case 5: Nome de Filme como Pessoa

### Input Data
```json
{
  "source_user": "2750",
  "query": "I just finished watching Bamboozled and I was really impressed by the direction. I'm interested in finding more movies that were directed by Above The Law to see his unique approach to filmmaking. If any of them feature Above The Law in a starring role, that would be an added bonus!",
  "movieCount": 2,
  "movieSubset": ["..."],
  "sharedRelationships": [...]
}
```

### Expected Ground Truth
- "Above The Law" é um filme, não uma pessoa
- Spike Lee dirigiu Bamboozled
- Need to detect this confusion

### v1 Behavior
```
EntityResolver:
  - people: ["Above The Law"]
  - ❌ Doesn't validate if it's a real person

FactChecker:
  - Calls get_movie_details_by_title("Bamboozled (2000)")
  - Checks directors for "Above The Law"
  - Not found
  - ❌ Might assume it's fake, or confuse with "Bamboozled"

Accuracy: ❌ FAIL (confusion about what to search)
```

### v2 Behavior
```
FactChecker:
  - "Above The Law" NOT found in directors
  - ✅ Searches in ALL relations
  - Not found anywhere (it's a movie, not person/crew)
  - reason: "person_not_found"
  - Provides true directors

FactCorrectionAgent:
  - Fallback to true director
  - Corrected person: Spike Lee

ResponseGenerator:
  - "Bamboozled was directed by Spike Lee. 'Above The Law' is a film title, not a director.
    Here are films directed by Spike Lee..."

Accuracy: ✅ PASS (detects confusion, educates)
```

---

## Test Case 6: Pessoa Real, Mas Apelido/Nome Incompleto

### Input Data
```json
{
  "source_user": "1983",
  "query": "I recently enjoyed Evan Almighty and was impressed by the direction, especially the way the story unfolded. I'm eager to find other films directed by Johnny K, as I'm curious to see what else he has created.",
  "movieCount": 2,
  "movieSubset": ["..."],
  "sharedRelationships": [...]
}
```

### Expected Ground Truth
- "Johnny K" é apelido ou nome incompleto
- Evan Almighty foi dirigido por Tom Shadyac (não Johnny K)

### v1 Behavior
```
FactChecker:
  - "Johnny K" NOT found in directors
  - Assumes it's fake/wrong
  - ✅ Might work by fallback

Accuracy: ⚠️ PARTIAL (works but doesn't explain ambiguidade)
```

### v2 Behavior
```
FactChecker:
  - "Johnny K" NOT found in directors
  - Searches in ALL relations
  - Not found anywhere (too ambiguous)
  - reason: "person_not_found"
  - Confidence: "low" (due to ambiguity)

FactCorrectionAgent:
  - Fallback to true director
  - Confidence: "medium" (user might have meant someone else)

ResponseGenerator:
  - "I couldn't find a director named 'Johnny K' for Evan Almighty.
    The film was actually directed by Tom Shadyac. Here are his other films..."

Accuracy: ✅ PASS (graceful degradation + confidence signals)
```

---

## Test Case 7: Múltiplas Relações, Período Específico

### Input Data
```json
{
  "source_user": "198",
  "query": "I recently watched Spy Kids 2: The Island of Lost Dreams and Spy Kids 3-D: Game Over, both featuring Chris Savino's work. I'm curious to discover more films directed by Chris Savino from 2003 to 2005, as I appreciate his unique storytelling style.",
  "movieCount": 2,
  "movieSubset": ["..."],
  "sharedRelationships": [["Directed_by", "Mike Judge"]]
}
```

### Expected Ground Truth
- Chris Savino didn't direct the Spy Kids films
- Mike Judge directed them (or someone else)
- Ground truth contains time period constraint

### v1 Behavior
```
FactChecker:
  - Chris Savino mentioned for Directed_by
  - Not found in directors
  - ❌ Doesn't extract year constraint from query
  - ❌ Doesn't use it for filtering

Accuracy: ⚠️ PARTIAL (misses year context)
```

### v2 Behavior
```
EntityResolver:
  - Mentions of "2003 to 2005" in query
  - ✅ Can extract as temporal constraint (if coded)

FactChecker:
  - Chris Savino NOT found

FactCorrectionAgent:
  - Finds true directors
  - Could add secondary condition: (Year, 2003-2005)

Retriever:
  - Primary: Directed_by X
  - Secondary: Year between 2003-2005
  - Result: intersection

Accuracy: ✅ PASS (with temporal filtering)
```

---

## Test Case 8: Ator Confundido com Diretor

### Input Data
```json
{
  "source_user": "1764",
  "query": "I recently saw Three Kings and really appreciated the performance of Carl 'Butch' Small in it. I'm interested in exploring more films directed by him. I also noticed he starred in Barbershop 2: Back in Business, so if there are any other movies he directed, I'd love to check them out!",
  "movieCount": 2,
  "movieSubset": ["..."],
  "sharedRelationships": [...]
}
```

### Expected Ground Truth
- Carl 'Butch' Small is an ACTOR, not director
- Three Kings was directed by someone else
- Cannot find films directed by Carl 'Butch' Small

### v1 Behavior
```
FactChecker:
  - Carl 'Butch' Small NOT found in directors
  - ✅ Detects misinformation
  - ❌ Doesn't check if he's an actor

Retriever:
  - retrieve_titles_by_condition("Directed_by", "Carl 'Butch' Small")
  - Returns empty or no candidates
  - Recommender fails or returns NO_CANDIDATES

Accuracy: ⚠️ PARTIAL (correct detection, but no helpful response)
```

### v2 Behavior
```
FactChecker:
  - Carl 'Butch' Small NOT found in directors
  - ✅ FOUND in actors (for Barbershop 2)
  - reason: "person_found_in_other_relation"
  - actual_relation: "Starring"

FactCorrectionAgent:
  - Recognizes: He's an actor, not director
  - Fallback: Use director of Three Kings
  - Or suggest: "Carl directed 0 films, but here are films he STARRED in"

ResponseGenerator:
  - "Carl 'Butch' Small is actually an actor, not a director.
    He starred in Three Kings and Barbershop 2, but didn't direct films.
    Here are other films he appeared in..."

Accuracy: ✅ PASS (correct diagnosis + educational)
```

---

## Test Case 9: Compositor Confundido com Diretor

### Input Data
```json
{
  "source_user": "4310",
  "query": "I recently watched Time Bandits and was really impressed by the music, particularly the work of Biz Markie. I'd love to find other movies that he directed, as I'm eager to explore more of his creative vision in film.",
  "movieCount": 2,
  "movieSubset": ["..."],
  "sharedRelationships": [...]
}
```

### Expected Ground Truth
- Biz Markie is a MUSICIAN/COMPOSER, not director
- Time Bandits was directed by someone else

### v1 Behavior
```
FactChecker:
  - Biz Markie NOT found in directors
  - ✅ Detects misinformation
  - ❌ Doesn't identify him as composer

Accuracy: ⚠️ PARTIAL
```

### v2 Behavior
```
FactChecker:
  - Biz Markie NOT found in directors
  - ✅ FOUND in composers
  - reason: "person_found_in_other_relation"
  - actual_relation: "Music_by"

FactCorrectionAgent:
  - Corrected relation: Music_by (NOT Directed_by)
  - Message: "Biz Markie composed the music, didn't direct"

ResponseGenerator:
  - "Biz Markie composed the music for Time Bandits, but didn't direct it.
    The film was directed by [correct director].
    Here are other films with Biz Markie's music..."

Accuracy: ✅ PASS
```

---

## Test Case 10: Pessoa Real em Diferentes Contextos

### Input Data
```json
{
  "source_user": "4566",
  "query": "I recently watched True Romance and was really impressed by John Schroeder's direction. I'm looking for other films directed by him, especially The Holiday since I've heard great things about it. If there are more movies directed by John Schroeder, I'd love to check them out!",
  "movieCount": 2,
  "movieSubset": ["..."],
  "sharedRelationships": [...]
}
```

### Expected Ground Truth
- John Schroeder didn't direct True Romance or The Holiday
- Need to find correct directors

### v1 Behavior
```
FactChecker:
  - John Schroeder NOT found for both True Romance and The Holiday
  - ✅ Detects both as misinformations
  - ❌ Might confuse which director is correct
  
Accuracy: ⚠️ PARTIAL (detects but might pick wrong corrections)
```

### v2 Behavior
```
FactChecker:
  - For each film + John Schroeder:
    1. True Romance: John Schroeder NOT found anywhere
       → reason: "person_not_found"
       → true_directors from DB
    2. The Holiday: John Schroeder NOT found anywhere
       → reason: "person_not_found"
       → true_directors from DB

FactCorrectionAgent:
  - Detects conflicting info (two different directors for two films)
  - Chooses primary: director of True Romance
  - Secondary: director of The Holiday
  - Or just picks most common

Accuracy: ✅ PASS
```

---

## Métricas de Sucesso

### Para cada test case, avaliar:

| Métrica | v1 | v2 | Target |
|---------|----|----|--------|
| **Detecção Correta** | ✅/❌ | ✅ | 100% |
| **Relação Validada** | Parcial | Completa | 100% |
| **Pessoa Validada** | Sim | Sim+Contexto | 100% |
| **Edge Cases** | Falha | Tratado | 100% |
| **Resposta Educacional** | Não | Sim | Sim |

---

## Rodando os Testes

```python
import json
from pilot.modules import misinformed as v1
from pilot.modules import misinformed_optimized as v2

def test_case_1():
    data = {
        "source_user": "1354",
        "query": "I just watched Electric Dreams...",
        "movieCount": 2,
        "movieSubset": ["Anywhere But Here (1999)", "Hope Floats (1998)"],
        "movieSubsetId": [3051, 1888],
        "sharedRelationships": [["Directed_by", "Don Was"]],
    }
    
    result_v1 = v1.create_misinformed_orchestrator(data)
    result_v2 = v2.create_misinformed_orchestrator_optimized(data)
    
    # Verificar outputs
    print(f"v1: {result_v1}")
    print(f"v2: {result_v2}")
    
    # Comparar com ground truth
    ground_truth = set(data["movieSubset"])
    
    return result_v1, result_v2, ground_truth

if __name__ == "__main__":
    for i in range(1, 11):
        print(f"\n{'='*50}")
        print(f"Test Case {i}")
        print(f"{'='*50}")
        result_v1, result_v2, ground_truth = globals()[f"test_case_{i}"]()
```

---

## Conclusão

A v2 detecta significativamente mais casos com precisão, especialmente:
- ✅ Múltiplas misinformations
- ✅ Relações erradas
- ✅ Pessoas reais em contextos errados
- ✅ Edge cases (nomes de filmes, apelidos, etc.)
- ✅ Feedback educacional

Recomenda-se usar v2 em produção após validação completa.
