# Advisor Design: Techniques for a Domain-Independent Conversational Mediator

## 1. The Problem the Advisor Solves

When users interact with multiple LLM-based recommenders, they face five empirically documented cognitive limitations that degrade decision quality.

**Information overload.** LLMs produce verbose, detailed responses. Across N systems, the user faces N x K items with explanations. Research consistently shows that decision quality drops — not rises — as information volume increases (Pothos et al., 2021; Roetzel, 2019). McKinsey's "Bias Busters" (2026) demonstrates that cognitive overload acts as a multiplier for all other biases.

**Choice overload.** Large recommendation sets reduce the probability of making any choice at all. Field experiments in online recommender systems show there is an optimal (small) list size — beyond it, conversion rate drops (MSOM, 2022). In LLM-specific contexts, "Decisions with ChatGPT" (2023) proves that choice overload exists in AI-mediated recommendations: when ChatGPT offers many options, user satisfaction decreases.

**Anchoring bias.** Users fixate disproportionately on the first item presented. Three controlled experiments (INFORMS, 2013) show that predicted ratings from a recommender system alter users' actual evaluations of items. In a multi-LLM setting, each system anchors the user to a different first item, compounding confusion.

**Confirmation bias and filter bubbles.** Each LLM reinforces its own framing of the user's query. A systematic review of filter bubbles in recommender systems (2024) shows that the interaction between user choices and system outputs creates feedback loops that progressively narrow the user's exposure. With multiple LLMs, each creates its own filter bubble.

**Bounded rationality and preference articulation difficulty.** Users cannot fully articulate what they want. "Interpretability Gone Bad" (Kaur et al., CSCW) demonstrates that even well-designed interpretability tools can worsen decisions under bounded rationality. Patient decision aid research (JMIR, 2016) shows that explicitly helping users articulate their preferences improves decision quality.

The Advisor mitigates these five limitations through conversational mediation, without domain expertise.

---

## 2. What the Advisor Is

The Advisor is a domain-independent conversational mediator. It observes outputs from N independent LLM recommenders and engages the user in a multi-turn dialogue to support decision-making. It has three — and only three — sources of information: the outputs of the N LLMs (items and explanations), what the user says during the conversation, and the interaction history (what has been shown, asked, and answered).

The Advisor is not a domain expert. It has no access to knowledge graphs, domain-specific databases, or curated metadata. It is not a recommender system — it does not generate recommendations. It is not a pre-filter — it does not silently modify queries before submission. Its expertise is in inquiry, comparison, and preference elicitation.

---

## 3. Techniques

### 3.1 Cross-System Triangulation

The Advisor compares outputs from N independent LLMs to detect consensus and divergence. At the item level, it computes set overlap (Jaccard similarity) between Top-K lists. At the explanation level, it measures semantic similarity between explanations of overlapping items using sentence embeddings. At the framing level, it detects when different LLMs interpret the same query through different lenses — one analytical, another personal, another regional.

Consensus provides a confidence signal: items recommended by multiple independent systems are more likely to be relevant. Divergence provides an exploration signal: where systems disagree, there is ambiguity worth investigating with the user. This cross-system analysis is structurally impossible for any individual LLM, which only sees its own output.

No existing work uses cross-LLM divergence as an information signal for conversational advisory. Ensemble methods aggregate outputs; the Advisor uses the pattern of agreement and disagreement as a basis for dialogue.

A known limitation: triangulation measures agreement, not truth. If all N LLMs hallucinate the same title, the Advisor will treat it as high-confidence. The Advisor provides "confidence based on agreement," not verified truth.

### 3.2 Contextual Bandit for Action Selection

At each conversational turn, the Advisor must decide what to do. This is a sequential decision problem that maps naturally to a contextual bandit formulation (drawing on EAR by Lei et al., WSDM 2020, and SCPR by Lei et al., NeurIPS 2020).

The context observed at each turn includes the turn number, the convergence score between LLM outputs, the specificity of the user's most recent response, the number of items already shown, whether divergence was detected, and the session number if the interaction spans multiple sessions.

The Advisor selects one action per turn from a discrete set: ask a preference-clarifying question, show a cross-system comparison, present top items by consensus with contrastive framing, highlight divergence and ask the user to resolve it, synthesize and ask for a decision, or end the session. A contextual bandit (LinUCB or Thompson Sampling) is appropriate because the number of turns per session is small (3–6); full reinforcement learning would require more data than is available.

Training the bandit requires either simulated users at scale or offline data from logged interactions. For a first experiment, a heuristic policy can be used to collect initial interaction logs, followed by offline bandit learning via inverse propensity scoring.

### 3.3 Belief State Tracking for User Modeling

The Advisor maintains a structured representation of the user's evolving preferences, updated at each turn. This draws on dialogue state tracking from task-oriented dialogue systems and on UNICORN (Deng et al., KDD 2021).

The user state is domain-independent and includes: preference specificity (a float from 0 to 1 indicating how specific the user's preference is), preference dimensions (a dictionary of extracted dimensions such as tone, pace, complexity, or formality), the set of items already seen, items the user reacted to (positively or negatively), an anchoring risk estimate (high if the user fixated on the first item shown), and an overload risk estimate (high if the user has seen many items without deciding).

At each turn, the Advisor uses a deterministic LLM call (temperature=0) to parse the user's response into structured updates to this state. This is the Advisor's "learning quickly" capability — it does not need prior domain knowledge to extract preference signals from natural language.

### 3.4 Preference Elicitation via Information Gain

When the Advisor asks a question, it should be the question that maximally reduces uncertainty about the user's preferences. This draws on active learning for preference elicitation and on the patient decision aid literature (JMIR, 2016), which shows that explicitly surfacing preference dimensions improves decision quality.

The Advisor approximates information gain using the diversity of LLM responses as a proxy for uncertainty. Where LLMs diverge most, there is the most uncertainty about what the user actually wants. A question that resolves this divergence has high expected information gain. This heuristic avoids the need for a full probabilistic model of preferences while remaining principled.

### 3.5 Debiasing Through Presentation Design

The Advisor's presentation strategy is explicitly designed to counter the documented biases, with each design decision backed by empirical evidence.

To counter anchoring, the Advisor never presents items in the order of any single LLM. Items are re-ordered by consensus score or presented as unordered paths. To counter choice overload, the Advisor presents a maximum of two to three options per turn, following the empirical finding that even three options can be excessive for online decision-making (UCLA field study). To counter confirmation bias, the Advisor always surfaces at least one divergent option — something recommended by only one LLM but matching the user's stated preferences — ensuring exposure to perspectives beyond the dominant framing. To counter information overload, the Advisor synthesizes explanations into one sentence per item rather than concatenating full LLM outputs.

### 3.6 Conversational Preference Refinement

The core mechanism through which the user becomes a better decision maker is a multi-turn dialogue cycle. At each turn, the Advisor presents information (a comparison, a set of options, or a question). The user responds. The Advisor extracts a structured preference update from the response. The user state is updated. The next action is selected based on the updated state.

Over turns, the user's preference specificity increases. The user moves from "I want a book about food" to "I want an emotional memoir about food and identity, with a connection to Italian culture." This refinement is the user becoming a better decision maker — they understand their own motivations and can articulate them. The Advisor facilitates this process without prescribing the outcome.

---

## 4. Measuring "Better Decision Maker"

### 4.1 Preference-Decision Alignment (primary metric)

At the end of a session, compare the experiential dimensions of the user's chosen item with the preference dimensions they articulated during the conversation. Compute cosine similarity between the two vectors. A user whose final choice aligns with their stated preferences is making a more informed decision than one who picks randomly or anchors on the first item.

### 4.2 Articulation Improvement (secondary metric)

Measure preference specificity at each turn and plot the trajectory. A user whose specificity increases from 0.2 to 0.8 over four turns has measurably improved their ability to express what they want.

### 4.3 Bias Resistance (secondary metric)

Measure whether the user resisted anchoring (final choice is not the first item shown), considered items from divergent LLMs (confirmation bias resistance), and completed the session with a decision rather than abandoning (choice overload resistance). These are proxy measures — a user may genuinely prefer the first item — but across a population, systematic patterns are informative.

### 4.4 Ground-Truth Alignment (where available)

For datasets with ground truth (RecBench+, InstructRec), compute Recall@K and Precision@K of the Advisor's final presented items. This measures recommendation quality, not decision quality directly, but provides a comparable baseline metric.

---

## 5. What the Advisor Does That LLMs Cannot

A single LLM sees only its own output. The Advisor compares outputs from multiple independent systems, detecting consensus and divergence that no individual system can observe.

LLMs generate long, detailed responses that overwhelm users. The Advisor synthesizes, selects, and sequences information to manage cognitive load, presenting focused options backed by cross-system evidence.

LLMs anchor users to their first recommendation. The Advisor re-orders by consensus and presents alternatives, breaking the anchoring pattern.

LLMs reinforce their own framing. The Advisor surfaces divergent perspectives, ensuring the user sees the full landscape of options rather than a single system's view.

LLMs cannot help users articulate what they want — they respond to what the user says. The Advisor asks targeted questions based on detected uncertainty, helping the user discover and refine their own preferences through dialogue.

---

## References

- Akata, Z., et al. (2020). A Research Agenda for Hybrid Intelligence. *IEEE Computer*, 53(8), 18–28.
- Saricks, J. G. (2005). *Readers' Advisory Service in the Public Library* (3rd ed.). ALA Editions.
- Pothos, E. M., et al. (2021). Information overload for (bounded) rational agents. *Proceedings of the Royal Society B*, 288(1944).
- Decisions with ChatGPT: Reexamining choice overload in AI-mediated recommendations. *Journal of Retailing and Consumer Services*, 2023.
- Do Recommender Systems Manipulate Consumer Preferences? A Study of Anchoring Effects. *Information Systems Research*, INFORMS, 2013.
- Filter Bubbles in Recommender Systems: Fact or Fallacy — A Systematic Review. 2024.
- The Choice Overload Effect in Online Recommender Systems. *Manufacturing & Service Operations Management*, 2022.
- Bias Busters: How cognitive overload multiplies every bias. McKinsey, 2026.
- Kaur, H., et al. Interpretability Gone Bad: The Role of Bounded Rationality. CSCW.
- Features of Computer-Based Decision Aids: Systematic Review. *JMIR*, 2016.
- Lei, W., et al. (2020). Estimation-Action-Reflection Framework for Conversational Recommendation. *WSDM*.
- Lei, W., et al. (2020). Interactive Path Reasoning on Graph for Conversational Recommendation. *NeurIPS*.
- Deng, Y., et al. (2021). Unified Conversational Recommendation Policy Learning via Graph-based Reinforcement Learning. *KDD*.
