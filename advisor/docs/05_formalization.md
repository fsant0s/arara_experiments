# Formalization: Advisor as a Contextual Bandit over Multi-LLM Recommendation

## 1. Problem Setting

A user \( u \) seeks recommendations in a domain \( \mathcal{D} \) (books, movies, restaurants, etc.). The user interacts with an Advisor \( \mathcal{A} \) over a session of \( T \) turns. The Advisor mediates between the user and \( N \) independent LLM recommenders \( \{\mathcal{R}_1, \ldots, \mathcal{R}_N\} \).

At the start of each session, the user submits a query \( q_u \) to all \( N \) recommenders. Each recommender \( \mathcal{R}_n \) returns a ranked list of \( K_n \) items with explanations:

\[
\mathcal{O}_n = \mathcal{R}_n(q_u) = \{(i_{n,1}, e_{n,1}), \ldots, (i_{n,K_n}, e_{n,K_n})\}
\]

where \( i_{n,k} \) is an item and \( e_{n,k} \) is its natural-language explanation from recommender \( n \).

The Advisor observes all \( N \) outputs and engages the user in a \( T \)-turn conversational session to help them make a decision.

---

## 2. State Space

### 2.1 User Belief State

At each turn \( t \), the Advisor maintains a structured representation of the user's evolving preferences:

\[
s_u^{(t)} = \left( \sigma^{(t)}, \; \mathbf{d}^{(t)}, \; \mathcal{I}_{\text{seen}}^{(t)}, \; \mathcal{I}_{\text{pos}}^{(t)}, \; \mathcal{I}_{\text{neg}}^{(t)}, \; \alpha^{(t)}, \; \omega^{(t)} \right)
\]

| Symbol | Type | Description |
|---|---|---|
| \( \sigma^{(t)} \) | \( [0, 1] \) | **Preference specificity**: how precisely the user has articulated their preference. Starts at \( \sigma^{(0)} \approx 0 \) for vague queries. |
| \( \mathbf{d}^{(t)} \) | \( \mathbb{R}^m \) | **Preference dimensions vector**: extracted experiential dimensions (e.g., tone, pace, complexity, formality). Initially sparse, populated as the user reveals preferences. Domain-independent: dimensions are abstract categories, not domain-specific attributes. |
| \( \mathcal{I}_{\text{seen}}^{(t)} \) | set | Items presented to the user by turn \( t \). |
| \( \mathcal{I}_{\text{pos}}^{(t)} \) | set | Items the user reacted to positively (expressed interest). |
| \( \mathcal{I}_{\text{neg}}^{(t)} \) | set | Items the user reacted to negatively (rejected or ignored). |
| \( \alpha^{(t)} \) | \( [0, 1] \) | **Anchoring risk**: estimated probability that the user is fixated on the first item shown. High if the user repeatedly references the first presented item without exploring alternatives. |
| \( \omega^{(t)} \) | \( [0, 1] \) | **Overload risk**: estimated probability that the user is overwhelmed. High if many items have been shown without a decision. |

### 2.2 System State

The Advisor also maintains system-level features computed from the LLM outputs:

\[
s_{\text{sys}}^{(t)} = \left( \gamma^{(t)}, \; \delta^{(t)}, \; t, \; |\mathcal{I}_{\text{seen}}^{(t)}| \right)
\]

| Symbol | Type | Description |
|---|---|---|
| \( \gamma^{(t)} \) | \( [0, 1] \) | **Convergence score**: Jaccard similarity of the Top-K item sets across \( N \) recommenders. \( \gamma = \frac{|\bigcap_n \text{Top-K}_n|}{|\bigcup_n \text{Top-K}_n|} \). |
| \( \delta^{(t)} \) | \( \{0, 1\} \) | **Divergence flag**: 1 if framing divergence is detected (LLMs interpret the query through substantially different lenses). |
| \( t \) | \( \mathbb{N} \) | Turn number. |
| \( |\mathcal{I}_{\text{seen}}^{(t)}| \) | \( \mathbb{N} \) | Number of items shown to user so far. |

### 2.3 Combined Context

The context vector for the bandit at turn \( t \) is the concatenation:

\[
\mathbf{x}^{(t)} = \left[ \sigma^{(t)}, \; \gamma^{(t)}, \; \delta^{(t)}, \; \alpha^{(t)}, \; \omega^{(t)}, \; t, \; |\mathcal{I}_{\text{seen}}^{(t)}| \right] \in \mathbb{R}^7
\]

---

## 3. Action Space

At each turn \( t \), the Advisor selects one action \( a^{(t)} \in \mathcal{A}_{\text{actions}} \):

| Action | Symbol | Description |
|---|---|---|
| Ask preference-clarifying question | \( a_1 \) | Select the question that maximally reduces uncertainty about \( \mathbf{d}^{(t)} \). Targets the dimension with highest entropy across LLM outputs. |
| Show cross-system comparison | \( a_2 \) | Present overlapping and divergent items across recommenders. Highlight what agrees and what differs. |
| Present top by consensus + contrastive framing | \( a_3 \) | Show the top items ranked by cross-system consensus (items in \( \bigcap_n \text{Top-K}_n \) first), with contrastive explanation: "why this and not that." |
| Highlight divergence, ask user to resolve | \( a_4 \) | Present a specific point of divergence between LLMs and ask the user which framing matches their intent. |
| Synthesize and ask for decision | \( a_5 \) | Summarize the session, present a structured choice, and ask the user to decide. |
| End session | \( a_6 \) | Close the session. Triggered when the user has made a decision or a maximum turn limit is reached. |

### 3.1 Action Constraints

- \( a_6 \) is forced when \( t = T_{\max} \) (maximum turns).
- \( a_6 \) is triggered when the user explicitly makes a decision.
- At most 2–3 items are presented per turn (choice overload constraint).
- Items are never presented in the order of any single LLM (anti-anchoring constraint).

---

## 4. Belief State Update

At each turn, after the user responds with utterance \( r_u^{(t)} \), the Advisor updates the belief state:

\[
s_u^{(t+1)} = \text{UPDATE}(s_u^{(t)}, r_u^{(t)})
\]

The UPDATE function is implemented as a deterministic LLM call (temperature = 0) that parses the user's natural-language response into structured updates:

```
Given:
  Current user state: s_u^(t)
  User response: r_u^(t)

Extract:
  - New preference dimensions mentioned (update d^(t))
  - Items referenced positively or negatively (update I_pos, I_neg)
  - Whether the user has fixated on a specific item (update α)
  - Whether the user seems overwhelmed or uncertain (update ω)
  - Updated preference specificity (update σ)

Rules:
  - Do not hallucinate dimensions the user did not mention.
  - σ increases only when the user provides new, specific information.
  - α increases if the user repeatedly references the same first-shown item.
  - ω increases if the user expresses confusion or asks to "go back."
```

### 4.1 Preference Specificity Update

Preference specificity \( \sigma^{(t)} \) is computed as:

\[
\sigma^{(t)} = \frac{|\{j : d_j^{(t)} \neq \text{unknown}\}|}{m}
\]

where \( m \) is the total number of possible experiential dimensions and \( d_j^{(t)} \) is the value of dimension \( j \) at turn \( t \). As the user articulates more dimensions, specificity increases monotonically.

---

## 5. Cross-System Triangulation

### 5.1 Item-Level Convergence

\[
\gamma_{\text{item}} = \frac{|\bigcap_{n=1}^{N} \mathcal{I}_n|}{|\bigcup_{n=1}^{N} \mathcal{I}_n|}
\]

where \( \mathcal{I}_n = \{i_{n,1}, \ldots, i_{n,K_n}\} \) is the item set from recommender \( n \).

### 5.2 Explanation-Level Convergence

For items appearing in multiple recommender outputs, compute semantic similarity between their explanations:

\[
\gamma_{\text{expl}}(i) = \frac{1}{\binom{N_i}{2}} \sum_{n < n'} \cos\left(\text{SBERT}(e_{n,i}), \; \text{SBERT}(e_{n',i})\right)
\]

where \( N_i \) is the number of recommenders that included item \( i \), and \( \text{SBERT}(\cdot) \) produces a sentence embedding.

### 5.3 Framing Divergence Detection

Framing divergence \( \delta \) is detected when the centroid embeddings of the explanation sets from different recommenders are dissimilar:

\[
\bar{e}_n = \frac{1}{K_n} \sum_{k=1}^{K_n} \text{SBERT}(e_{n,k})
\]

\[
\delta = \mathbb{1}\left[\min_{n \neq n'} \cos(\bar{e}_n, \bar{e}_{n'}) < \tau_{\text{frame}}\right]
\]

where \( \tau_{\text{frame}} \) is a threshold (e.g., 0.7). If the minimum pairwise cosine similarity between explanation centroids falls below this threshold, the LLMs are interpreting the query through substantially different lenses.

---

## 6. Contextual Bandit Algorithm

### 6.1 LinUCB Formulation

At each turn \( t \), the Advisor selects the action that maximizes an upper confidence bound on expected reward:

\[
a^{(t)} = \arg\max_{a \in \mathcal{A}_{\text{actions}}} \left( \hat{\boldsymbol{\theta}}_a^\top \mathbf{x}^{(t)} + \beta \sqrt{\mathbf{x}^{(t)\top} \mathbf{A}_a^{-1} \mathbf{x}^{(t)}} \right)
\]

where:
- \( \hat{\boldsymbol{\theta}}_a \in \mathbb{R}^7 \) is the learned parameter vector for action \( a \).
- \( \mathbf{A}_a \in \mathbb{R}^{7 \times 7} \) is the design matrix for action \( a \).
- \( \beta > 0 \) is the exploration parameter.

### 6.2 Parameter Update

After observing reward \( r^{(t)} \) (computed at session end and distributed to all turns):

\[
\mathbf{A}_{a^{(t)}} \leftarrow \mathbf{A}_{a^{(t)}} + \mathbf{x}^{(t)} \mathbf{x}^{(t)\top}
\]

\[
\mathbf{b}_{a^{(t)}} \leftarrow \mathbf{b}_{a^{(t)}} + r^{(t)} \mathbf{x}^{(t)}
\]

\[
\hat{\boldsymbol{\theta}}_{a^{(t)}} \leftarrow \mathbf{A}_{a^{(t)}}^{-1} \mathbf{b}_{a^{(t)}}
\]

### 6.3 Initialization (Heuristic Policy)

Before sufficient data is available for bandit learning, a heuristic policy provides the initial behavior:

| Condition | Action |
|---|---|
| \( \sigma^{(t)} < 0.3 \) | \( a_1 \) (ask clarifying question) |
| \( \sigma^{(t)} \geq 0.3 \) and \( \delta = 1 \) | \( a_4 \) (highlight divergence) |
| \( \sigma^{(t)} \geq 0.3 \) and \( \gamma > 0.5 \) | \( a_3 \) (present by consensus) |
| \( \sigma^{(t)} \geq 0.5 \) and \( t \geq 3 \) | \( a_5 \) (synthesize and ask for decision) |
| User decides | \( a_6 \) (end session) |

This heuristic policy generates logged interaction data for offline bandit learning via inverse propensity scoring.

---

## 7. Reward Function

The reward is computed at session end and reflects the degree to which the user became a "better decision maker."

### 7.1 Composite Reward

\[
r = w_1 \cdot r_{\text{align}} + w_2 \cdot r_{\text{artic}} + w_3 \cdot r_{\text{bias}} + w_4 \cdot r_{\text{gt}}
\]

where \( w_1 + w_2 + w_3 + w_4 = 1 \) and the components are:

### 7.2 Preference-Decision Alignment \( r_{\text{align}} \)

\[
r_{\text{align}} = \cos\left(\mathbf{d}^{(T)}, \; \mathbf{d}_{\text{chosen}}\right)
\]

where \( \mathbf{d}^{(T)} \) is the user's articulated preference vector at session end and \( \mathbf{d}_{\text{chosen}} \) is the experiential dimension vector of the chosen item. This measures whether the user's final choice is consistent with what they said they wanted.

### 7.3 Articulation Improvement \( r_{\text{artic}} \)

\[
r_{\text{artic}} = \sigma^{(T)} - \sigma^{(0)}
\]

The increase in preference specificity over the session. A user who moved from vague to specific has improved their ability to articulate what they want.

### 7.4 Bias Resistance \( r_{\text{bias}} \)

\[
r_{\text{bias}} = \frac{1}{3}\left( r_{\text{anchor}} + r_{\text{confirm}} + r_{\text{overload}} \right)
\]

where:

- \( r_{\text{anchor}} = \mathbb{1}[\text{chosen item} \neq \text{first item shown by any single LLM}] \)
- \( r_{\text{confirm}} = \frac{|\{n : \exists \, i \in \mathcal{I}_{\text{pos}}^{(T)} \cap \mathcal{I}_n\}|}{N} \) (fraction of LLMs whose items the user considered)
- \( r_{\text{overload}} = \mathbb{1}[\text{user made a decision before } T_{\max}] \)

### 7.5 Ground-Truth Alignment \( r_{\text{gt}} \) (where available)

\[
r_{\text{gt}} = \text{Recall@K}\left(\mathcal{I}_{\text{presented}}^{(T)}, \; \mathcal{I}_{\text{ground truth}}\right)
\]

Available only for datasets with ground-truth labels (RecBench+, InstructRec). Set \( w_4 = 0 \) when ground truth is unavailable.

---

## 8. Debiasing Constraints (Hard-Coded)

These constraints are not learned — they are architectural invariants that enforce debiasing regardless of the bandit's action selection.

| Constraint | Formal Description |
|---|---|
| **Anti-anchoring** | Items are never presented in the order \( (i_{n,1}, i_{n,2}, \ldots) \) of any single LLM \( n \). Re-ordering is by consensus score \( c(i) = |\{n : i \in \mathcal{I}_n\}| \) or random among equally scored items. |
| **Anti-choice-overload** | \( |\mathcal{I}_{\text{presented at turn } t}| \leq 3 \) for all \( t \). |
| **Anti-confirmation-bias** | At least one item from the "minority" recommendation (appearing in only one LLM's output) is included when items are presented, if it matches the user's stated preferences. |
| **Anti-information-overload** | Explanations are synthesized to one sentence per item. Raw LLM explanations are never concatenated or shown in full. |

---

## 9. Session Protocol

For each user \( u \):

1. User submits query \( q_u \).
2. All \( N \) recommenders return outputs \( \{\mathcal{O}_1, \ldots, \mathcal{O}_N\} \).
3. Advisor computes \( \gamma^{(0)} \), \( \delta^{(0)} \), initializes \( s_u^{(0)} \).
4. **For** \( t = 1 \) to \( T_{\max} \):
   a. Advisor constructs context \( \mathbf{x}^{(t)} \).
   b. Advisor selects action \( a^{(t)} \) via LinUCB (or heuristic policy).
   c. Advisor generates response following action \( a^{(t)} \) and debiasing constraints.
   d. User responds with \( r_u^{(t)} \).
   e. Advisor updates \( s_u^{(t+1)} = \text{UPDATE}(s_u^{(t)}, r_u^{(t)}) \).
   f. **If** user has decided **or** \( t = T_{\max} \): select \( a_6 \), go to step 5.
5. Compute reward \( r \).
6. Update bandit parameters for all actions taken in the session.

---

## 10. Experimental Conditions

### 10.1 Baseline: No Advisor

The user receives the raw concatenated outputs of all \( N \) LLMs without any mediation. No belief state tracking, no cross-system comparison, no debiasing constraints. The user must synthesize and decide on their own.

### 10.2 Heuristic Advisor

The Advisor operates with the heuristic policy (Section 6.3) and all debiasing constraints (Section 8). No learning occurs — the policy is fixed. This isolates the effect of the architectural design (triangulation, debiasing, preference elicitation) from the learned policy.

### 10.3 Learned Advisor (LinUCB)

The Advisor operates with the learned bandit policy after training on logged interactions from the Heuristic Advisor condition. This tests whether learning to select actions improves outcomes beyond the fixed heuristic.

### 10.4 Ablations

| Ablation | Description |
|---|---|
| No debiasing constraints | Bandit selects actions, but presentation order and item count are unconstrained. |
| No triangulation | Advisor acts on outputs from a single LLM (randomly selected), not all \( N \). |
| No preference elicitation | Actions \( a_1 \) and \( a_4 \) are removed; Advisor can only present items, not ask questions. |

---

## 11. Datasets and Configuration

### 11.1 InstructRec (Primary)

- **Domains**: Books, Movies, Reads, Yelp.
- **Query structure**: Persona + instruction. Instructions are naturally vague and persona-influenced — precisely the bounded rationality challenge.
- **Ads injection**: The `_1ads` version injects advertisement items with persuasive descriptions. Tests anchoring and manipulation resistance.
- **Use**: Primary evaluation dataset. Books for development, Yelp for domain generalization, ads version for bias resistance.

### 11.2 RecBench+ (Secondary)

- **Domains**: Books, Movies.
- **Query structure**: Explicit, Implicit, Misinformed. Use Explicit and Implicit only (Misinformed is unrealistic per professor feedback).
- **Ground truth**: Per-query item sets with cutoff K.
- **Use**: Enables \( r_{\text{gt}} \) computation. Tests whether the Advisor helps users get relevant items when queries are vague (Implicit).

### 11.3 Configuration

| Parameter | Value |
|---|---|
| \( N \) (number of LLMs) | 2–3 (GPT-4o, Gemini, Claude) |
| \( K \) (items per LLM) | Dataset-dependent (typically 3–10) |
| \( T_{\max} \) | 6 turns |
| \( m \) (preference dimensions) | 6 (intensity, complexity, familiarity, emotional register, formality, scope) |
| \( \beta \) (exploration parameter) | 0.5 (tuned via validation) |
| LLM temperature (Advisor) | 0 |
| LLM temperature (recommenders) | Default (system-dependent) |

---

## 12. Summary of Notation

| Symbol | Meaning |
|---|---|
| \( u \) | User |
| \( q_u \) | User query |
| \( \mathcal{R}_n \) | Recommender system \( n \) |
| \( \mathcal{O}_n \) | Output of recommender \( n \) |
| \( i_{n,k} \) | Item \( k \) from recommender \( n \) |
| \( e_{n,k} \) | Explanation of item \( k \) from recommender \( n \) |
| \( s_u^{(t)} \) | User belief state at turn \( t \) |
| \( \sigma^{(t)} \) | Preference specificity |
| \( \mathbf{d}^{(t)} \) | Preference dimensions vector |
| \( \alpha^{(t)} \) | Anchoring risk |
| \( \omega^{(t)} \) | Overload risk |
| \( \gamma^{(t)} \) | Cross-system convergence score |
| \( \delta^{(t)} \) | Framing divergence flag |
| \( \mathbf{x}^{(t)} \) | Context vector for bandit |
| \( a^{(t)} \) | Action selected at turn \( t \) |
| \( r \) | Session reward |
| \( T_{\max} \) | Maximum turns per session |
