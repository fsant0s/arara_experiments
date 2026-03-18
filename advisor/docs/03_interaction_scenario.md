# Interaction Scenario: Domain-Independent Conversational Advisor

## Context

- **Domain**: Books on food, culture, and identity.
- **User**: A student currently living in Bolzano, Italy, interested in exploring how food connects to cultural identity. No strong prior reading history in this topic.
- **Recommendation systems**: Two independent LLM recommenders — ChatGPT (GPT) and Gemini.
- **Advisor**: A domain-independent conversational mediator positioned on the user's side. The Advisor has no access to knowledge graphs or domain databases. Its only information sources are the LLM outputs, what the user says, and the interaction history.

This scenario demonstrates five conversational turns. At each turn, the Advisor's internal state is annotated: the belief state update, the contextual bandit's action selection, the cognitive biases being mitigated, and the measurement signals observed.

---

## Turn 1: Initial Query and Cross-System Triangulation

### User Query

> "I want to read a book about the relationship between food, culture, and identity."

### Advisor Internal State (Before Acting)

**Belief state:**

| Field | Value |
|---|---|
| `preference_specificity` | 0.15 (very vague — broad topic, no style/tone indication) |
| `preference_dimensions` | `{"topic": "food + culture + identity"}` |
| `items_seen` | ∅ |
| `items_reacted_to` | ∅ |
| `anchoring_risk` | 0.0 |
| `overload_risk` | 0.0 |

**Context vector for bandit:**

| Feature | Value |
|---|---|
| Turn number | 1 |
| Convergence score | Not yet computed |
| User preference clarity | Low (0.15) |
| Items shown | 0 |
| Divergence detected | Not yet assessed |

### LLM Outputs

**GPT** returned 5 recommendations: *The Omnivore's Dilemma* (Pollan), *Taste of War* (Collingham), *Crying in H Mart* (Zauner), *The Language of Baklava* (Abu-Jaber), *Gastropolis* (Hauck-Lawson & Deutsch). GPT asked: "Would you prefer something more academic, narrative, or philosophical?"

**Gemini** returned 6 recommendations: *Food Identity* (Sofia Marquez, 2026), *The Cooking Gene* (Twitty), *Sweetness and Power* (Mintz), *Delizia!* (Dickie), *Crying in H Mart* (Zauner), *Food, Drink and Identity* (Scholliers, ed.). Gemini asked: "Would you like me to find a local bookstore in Bolzano?"

### Advisor Analysis

**Cross-system triangulation:**
- Total unique titles: 10. Overlap: 1 title — *Crying in H Mart*.
- Item-level Jaccard: 1/10 = 0.10 (very low convergence).
- Framing divergence detected: GPT anchors on "food systems" (agriculture, economics, geopolitics). Gemini anchors on "personal and cultural-historical identity" (ancestry, regional history, cultural psychology).
- This framing divergence is a signal that the user's query is ambiguous — both interpretations are valid, but they lead to fundamentally different reading experiences.

**Cognitive bias assessment:**
- *Information overload risk*: 10 unique titles with full explanations across 2 systems. Presenting all would degrade decision quality.
- *Anchoring risk*: If titles are presented in order, the user will fixate on whatever appears first. GPT's first item (analytical) vs. Gemini's first item (psychological) would anchor the user in opposite directions.
- *Confirmation bias risk*: Low at this stage (no prior interaction to confirm).

**Consistency check (triangulation):**
- *Food Identity* by Sofia Marquez (2026) appears only in Gemini's output. No cross-system corroboration. Flagged as potentially hallucinated.

**Bandit action selection:**
- Context: Turn 1, low convergence (0.10), low preference clarity (0.15), 0 items shown, divergence detected.
- Selected action: **A1 — Ask preference-clarifying question.** Rationale: preference clarity is too low to present items meaningfully. The detected framing divergence provides a natural basis for the question.

### Advisor Response to User

> **Advisor**: I consulted two independent recommendation systems on "food, culture, and identity." Here is what I found.
>
> Both systems agree on one title: ***Crying in H Mart*** by Michelle Zauner — a memoir about Korean-American identity explored through food and grief. The fact that two independent systems highlight this book increases confidence that it matches your interest.
>
> Beyond that, the systems diverge in an informative way. One leans toward **systemic and analytical** perspectives — how food shapes economies, politics, and social structures. The other leans toward **personal and cultural-historical** narratives — how food carries identity through ancestry, migration, and regional pride.
>
> Before I go deeper, I want to understand your reading preference. This will help me narrow down which direction suits you:
>
> **A)** Something that reads like a **personal story** — intimate, emotional, memoir-like.
> **B)** Something more **analytical** — examining food as a system, with historical and economic arguments.
> **C)** Something **region-specific** — exploring the food culture of a particular place.
>
> *One note: a title called "Food Identity" by Sofia Marquez could not be corroborated across systems and may not exist. I will focus on verified titles.*

**Cognitive biases mitigated:**

| Bias | How mitigated |
|---|---|
| Information overload | Reduced 10 titles to a structured summary with 2 directions + 1 consensus title |
| Anchoring | Did not present any list in order; framed as "paths" rather than ranked items |
| Confirmation bias | Exposed the user to both framings (analytical and personal) rather than defaulting to one |

**Measurement signals:**

| Metric | Observation |
|---|---|
| Preference-Decision Alignment | Cannot yet compute (no preference articulated beyond topic) |
| Articulation Improvement | Baseline: `preference_specificity` = 0.15 |
| Bias Resistance | Anchoring countermeasure applied; user has not yet been exposed to ordered items |

---

## Turn 2: Preference Clarification

### User Response

> "I prefer something that reads like a story."

### Advisor Internal State (Update)

**Belief state update:**

| Field | Previous | Updated |
|---|---|---|
| `preference_specificity` | 0.15 | 0.40 (style preference now known: narrative/memoir) |
| `preference_dimensions` | `{"topic": "food+culture+identity"}` | `{"topic": "food+culture+identity", "style": "narrative", "tone": "personal"}` |
| `anchoring_risk` | 0.0 | 0.0 (no items shown yet in ranked form) |
| `overload_risk` | 0.0 | 0.0 |

**Context vector for bandit:**

| Feature | Value |
|---|---|
| Turn number | 2 |
| Convergence score | 0.10 (from Turn 1) |
| User preference clarity | Medium (0.40) |
| Items shown | 0 (paths described, but no specific items presented for decision) |
| Divergence detected | Yes (framing divergence) |

### Advisor Action

The Advisor crafts different comparison queries for each LLM based on what each originally recommended, maximizing information gain:

- **To GPT**: "Between *Crying in H Mart* and *The Language of Baklava*, which is more emotionally engaging as a narrative?"
- **To Gemini**: "Between *Crying in H Mart* and *The Cooking Gene*, which is more narrative and story-like?"

### LLM Outputs

**GPT** compared *Crying in H Mart* vs. *The Language of Baklava*: Zauner's book has higher emotional intensity, intimate tone, focused on one relationship (mother-daughter) and grief. Abu-Jaber's is warmer, nostalgic, broader (family ensemble, coming-of-age). GPT's verdict: *Crying in H Mart* is stronger for emotional, personal storytelling.

**Gemini** compared *Crying in H Mart* vs. *The Cooking Gene*: Zauner's is a pure memoir with emotional pacing — "like sitting with a friend." Twitty's is a historical detective story that frequently pauses for historical context — requires more "academic energy." Gemini's verdict: *Crying in H Mart* reads like a story; *The Cooking Gene* does not.

### Advisor Analysis

**Cross-system triangulation (Turn 2):**
- Convergence increased: both LLMs independently confirm *Crying in H Mart* as the strongest narrative match against different comparators. Updated convergence for this item: high.

**Information gain assessment:**
- The comparison queries resolved the key uncertainty: *which* narrative-style books actually read as narratives vs. which are merely classified as memoirs but read academically.
- The Advisor now knows that *The Language of Baklava* is a softer alternative and *The Cooking Gene* requires academic energy. Both are viable but secondary.

**Contextual opportunity detected:**
- The user lives in Italy. Gemini's original list included *Delizia!* — a book about Italian food identity. This was not part of the narrative comparison but represents unexplored territory aligned with the user's life context. Surfacing this would address *bounded rationality*: the user did not know to ask about this dimension.

**Bandit action selection:**
- Context: Turn 2, high convergence on one item, medium preference clarity (0.40), 0 items formally presented for decision, divergence on secondary items.
- Selected action: **A3 — Present top items by consensus with contrastive framing.** Rationale: convergence is strong enough to present a primary recommendation. Adding the contextual suggestion (Delizia!) as a proactive follow-up addresses preference dimensions the user has not yet articulated.

### Advisor Response to User

> **Advisor**: Both systems strongly agree. ***Crying in H Mart*** by Michelle Zauner is the clear match for what you described.
>
> Here is how it reads: it is a memoir with fast emotional pacing, centered on one deep relationship — the author and her mother. The tone is intimate and urgent. Zauner writes about the specific smell of fermented soybean paste and the way her mother cut fruit. It is sensory, visceral, and personal.
>
> Both systems also offered alternatives — *The Language of Baklava* (warmer, nostalgic, broader family story) and *The Cooking Gene* (personal quest mixed with dense history). For pure narrative immersion, *Crying in H Mart* stands above both.
>
> One additional thought: since you are living in **Italy**, there is a book from the initial recommendations worth exploring — ***Delizia! The Epic History of the Italians and Their Food*** by John Dickie. It is a different kind of book (more intellectual and historical), but it would connect directly to the food culture around you. Would you like me to ask both systems about it?

**Cognitive biases mitigated:**

| Bias | How mitigated |
|---|---|
| Anchoring | Primary recommendation is based on cross-system consensus, not the order of any single LLM |
| Choice overload | Presented 1 primary + 2 brief alternatives + 1 optional exploration — not the full list |
| Bounded rationality | Proactively surfaced the Italian dimension the user did not think to request |
| Information overload | Synthesized two detailed LLM comparisons into focused appeal-based descriptions |

**Measurement signals:**

| Metric | Observation |
|---|---|
| Articulation Improvement | `preference_specificity` increased from 0.15 → 0.40 |
| Anchoring Resistance | User has not yet been exposed to any ranked list; primary recommendation based on consensus |
| Preference-Decision Alignment | User's stated "narrative/personal" aligns with *Crying in H Mart*'s profile |

---

## Turn 3: Exploring the Italian Angle

### User Response

> "Yes, tell me more about Delizia! and how it compares."

### Advisor Internal State (Update)

**Belief state update:**

| Field | Previous | Updated |
|---|---|---|
| `preference_specificity` | 0.40 | 0.50 (user expressed interest in a second dimension: regional/Italian) |
| `preference_dimensions` | `{"topic":..., "style": "narrative", "tone": "personal"}` | `{"topic":..., "style": "narrative", "tone": "personal", "context": "Italian/regional"}` |
| `items_seen` | ∅ | `{Crying in H Mart, The Language of Baklava, The Cooking Gene}` (described, not formally presented) |
| `items_reacted_to` | ∅ | `{"Delizia!": "curious"}` |
| `overload_risk` | 0.0 | 0.15 (user has now been exposed to 4+ titles; monitoring) |

**Bandit action selection:**
- Context: Turn 3, convergence high on primary item, medium-high clarity (0.50), 3 items described, user initiated a new exploration direction.
- Selected action: **A4 — Highlight divergence and ask user to resolve.** Rationale: the user is now exploring a second dimension (Italian culture). Querying both LLMs and highlighting how they differ on this dimension will help the user understand their own preference.

### Advisor Action

Different queries to maximize information gain:
- **To GPT**: "How does *Delizia!* compare to *Crying in H Mart* in terms of reading experience?"
- **To Gemini**: "Is *Delizia!* the best book to understand Italian food culture and regional identity? How does it compare to other Italian food books?"

### LLM Outputs

**GPT** provided a direct comparison: *Delizia!* is cultural history (lower emotional intensity, educational feel), while *Crying in H Mart* is a deeply personal memoir (higher intensity, intimate feel). GPT noted: "Because you are living in Italy, *Delizia!* might spark 'aha' moments about the food around you."

**Gemini** expanded the scope, surfacing 3 additional Italian food culture books: *Why Italians Love to Talk About Food* (Kostioukovitch — organized by region, warmer tone), *The Silver Spoon* (practical cooking reference), and *National Rice* (Bianchi, 2025 — North/South divide through starch). Gemini argued Kostioukovitch's book might feel more "personal" than *Delizia!* for understanding regional food identity.

### Advisor Analysis

**Cross-system triangulation (Turn 3):**
- New divergence: GPT stayed focused on the comparison the user asked for. Gemini expanded the Italian landscape with 3 new titles.
- This divergence reveals new options the user did not know about. It is productive divergence.

**Consistency check:**
- *National Rice* by Marco Bianchi (2025) appears only in Gemini. Unverifiable — flagged.

**Cognitive risk update:**
- *Overload risk* is rising: the user has now been exposed to multiple titles across two dimensions (narrative + Italian). The Advisor should structure rather than expand.

**Bandit action reflection:**
- The divergence between GPT (focused comparison) and Gemini (expanded landscape) provides a natural structure: present the user's emerging choice as two paths rather than an open-ended list.

### Advisor Response to User

> **Advisor**: Both systems confirm ***Delizia!*** as a strong book — but they illuminate it differently.
>
> From the direct comparison: *Delizia!* is a fundamentally different reading experience from *Crying in H Mart*. Where Zauner gives emotional intimacy, Dickie gives intellectual discovery. His argument: "Italian cuisine" does not exist — only the cuisines of its cities do. It is witty, fast-paced for a history book, and full of surprising stories.
>
> One system also surfaced an alternative: ***Why Italians Love to Talk About Food*** by Elena Kostioukovitch. It is organized by region — you could look up Trentino-Alto Adige (where Bolzano is) and understand the specific food culture around you. It is warmer and more descriptive than *Delizia!*, which is more argumentative.
>
> At this point, **two clear paths** have emerged:
>
> | Path | Book(s) | What you get |
> |---|---|---|
> | **Emotional / Narrative** | *Crying in H Mart* | A personal journey through food, grief, and identity — intimate and urgent |
> | **Cultural / Intellectual** | *Delizia!* or *Kostioukovitch* | Understanding how food built Italian identity — witty and revelatory, or warm and regional |
>
> These are not competing — they serve different reading moods. You could start with one and follow with the other.
>
> *Note: a title called "National Rice" (Bianchi, 2025) could not be verified across systems. I will focus on established titles.*
>
> Would you like to explore either path further, or do you feel ready to decide?

**Cognitive biases mitigated:**

| Bias | How mitigated |
|---|---|
| Choice overload | Structured the expanding information into exactly 2 paths with 3 total books |
| Anchoring | Paths are presented as unordered and explicitly described as "serving different moods" |
| Information overload | Synthesized Gemini's expanded output (4 new titles) into 1 relevant addition; deprioritized *The Silver Spoon* and *National Rice* |
| Confirmation bias | Surfaced a divergent option (Kostioukovitch) that neither system strongly pushed |

**Measurement signals:**

| Metric | Observation |
|---|---|
| Articulation Improvement | `preference_specificity` increased from 0.40 → 0.50 (new dimension: Italian/regional) |
| Overload risk | Rising (0.15). Advisor is structuring rather than expanding. |
| Bias Resistance | User is considering items from both LLMs and across two framings |

---

## Turn 4: User Decision

### User Response

> "I think I will go with *Crying in H Mart* first, but I am curious about the Italian books too."

### Advisor Internal State (Update)

**Belief state update:**

| Field | Previous | Updated |
|---|---|---|
| `preference_specificity` | 0.50 | 0.75 (user has made a decision and expressed secondary interest) |
| `items_reacted_to` | `{"Delizia!": "curious"}` | `{"Crying in H Mart": "chosen", "Delizia!": "interested", "Kostioukovitch": "interested"}` |
| `overload_risk` | 0.15 | 0.0 (user has decided; overload resolved) |
| `anchoring_risk` | 0.0 | Low — user's choice aligns with cross-system consensus, not presentation order |

**Bandit action selection:**
- Context: Turn 4, user has made a decision, preference clarity high (0.75), expressed secondary interest.
- Selected action: **A5 — Synthesize and suggest follow-up**, transitioning toward **A6 — End session.**

### Advisor Response to User

> **Advisor**: Here is a summary of what we explored and a reading plan.
>
> **Your primary pick: *Crying in H Mart* by Michelle Zauner.**
> Both systems independently converged on this title as the best match for your stated preference — a personal, narrative book about food and identity. It is a memoir centered on a daughter's relationship with her Korean mother, told through food. Fast emotional pacing, deeply intimate, visceral prose.
>
> **Your follow-up: Italian food culture.**
> Two options supported across systems:
> - ***Delizia!*** (Dickie) — a witty, argumentative history of Italy told through its cities. Investigative storytelling.
> - ***Why Italians Love to Talk About Food*** (Kostioukovitch) — a warm, region-by-region cultural guide. A companion for living in Italy.
>
> **Session summary:**
>
> | Aspect | Finding |
> |---|---|
> | **Convergence** | Both systems agreed on *Crying in H Mart* for narrative/personal reading |
> | **Productive divergence** | Systems disagreed on framing (analytical vs. personal), which helped map the full landscape |
> | **New discovery** | The Italian food culture dimension was not in the original query — it emerged through exploration |
> | **Flagged risks** | Two titles (*Food Identity*, *National Rice*) could not be verified and were deprioritized |
>
> If you finish *Crying in H Mart* and want a similar reading experience, *The Language of Baklava* by Diana Abu-Jaber would be a natural next step — warmer, broader, but in the same personal-food-identity space.

**Cognitive biases mitigated:**

| Bias | How mitigated |
|---|---|
| Post-decision regret / information overload | Clean summary reduces residual uncertainty |
| Anchoring | Final choice was driven by cross-system consensus, not presentation order |
| Bounded rationality | User discovered a dimension (Italian food culture) they had not thought to request |

---

## Turn 5: Closing and Reflection

### Advisor Final Synthesis

At session end, the Advisor computes final measurement signals:

**Preference-Decision Alignment:**
- User's articulated preferences: `{"style": "narrative", "tone": "personal", "topic": "food+culture+identity"}`.
- Chosen item (*Crying in H Mart*) profile: `{"style": "memoir", "tone": "intimate/personal", "topic": "food+grief+Korean-American identity"}`.
- Cosine similarity between preference vector and item vector: **high alignment**. The user chose an item consistent with what they said they wanted.

**Articulation Improvement:**
- `preference_specificity` trajectory: 0.15 → 0.40 → 0.50 → 0.75.
- The user moved from "food, culture, and identity" (vague topic) to a clear articulation of wanting an emotional memoir with a secondary interest in Italian regional food culture. This trajectory is the measurable improvement in decision-making capability.

**Bias Resistance:**
- *Anchoring*: The user's final choice was not the first item of either LLM (GPT's first item was *The Omnivore's Dilemma*; Gemini's was *Food Identity*). Anchoring was resisted.
- *Confirmation bias*: The user considered items from both framings (personal and cultural-historical) and across both LLMs. No single-system filter bubble formed.
- *Choice overload*: The user made a clear decision with a secondary follow-up plan, despite being exposed to 10+ unique titles over the session. Overload was managed.
- *Information overload*: At no point did the user receive the raw, unfiltered output of both LLMs. Information was always synthesized and structured.

**Contextual Bandit Action Trace:**

| Turn | Context Summary | Action Selected | Outcome |
|---|---|---|---|
| 1 | Low convergence, low preference clarity, divergence detected | A1: Ask preference-clarifying question | User articulated style preference |
| 2 | High convergence on 1 item, medium clarity | A3: Present top by consensus + contrastive framing | User engaged with primary pick + expressed curiosity about Italian angle |
| 3 | User exploring new dimension, overload risk rising | A4: Highlight divergence, ask user to resolve | User ready to decide with 2 clear paths |
| 4 | User decided, secondary interest expressed | A5: Synthesize + A6: End session | Session closed with decision + follow-up plan |

---

## Comparison: With Advisor vs. Without Advisor

| Dimension | Without Advisor | With Advisor |
|---|---|---|
| **Information presented** | 10 titles with full explanations from 2 LLMs | 2–3 focused options per turn, structured by reading experience |
| **Convergence visibility** | User must manually compare two long lists | Advisor highlights cross-system agreement as a confidence signal |
| **Divergence visibility** | User might not notice different framings | Advisor uses framing divergence as a basis for preference exploration |
| **Anchoring risk** | User sees each LLM's ranked list; anchors on first items | Advisor re-orders by consensus; presents paths, not ranked lists |
| **Choice overload** | 10+ options with no structure | Maximum 2–3 options per turn; structured into paths |
| **Hallucination risk** | User might pursue unverifiable titles | Advisor flags 2 titles lacking cross-system corroboration |
| **Preference discovery** | User gets only what they asked for | Advisor proactively surfaces Italian dimension based on life context |
| **Articulation** | User's preference remains vague ("food and identity") | User's preference refines to "emotional memoir" + "Italian regional culture" |
| **Decision quality** | Unknown — no structured support | Measurable via preference-decision alignment, articulation trajectory, bias resistance |
