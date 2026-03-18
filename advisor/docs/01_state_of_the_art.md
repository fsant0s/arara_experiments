# State of the Art: LLM-Based Advisory Systems for Conversational Recommendation

## 1. Scope

This review surveys work at the intersection of three research areas: (a) LLM-powered agents that mediate between users and recommendation systems, (b) empirically documented cognitive limitations that degrade human decision-making when navigating complex information, and (c) techniques for conversational preference elicitation and decision support. The focus is on identifying what existing systems address and what gap remains for a domain-independent conversational Advisor that mitigates cognitive biases in multi-LLM recommendation settings.

---

## 2. Empirically Documented Cognitive Limitations

The following five limitations are established in the literature and directly relevant to users navigating LLM-generated recommendations.

### 2.1 Information Overload

The core finding is robust: beyond a threshold, more information degrades rather than improves decision quality. Pothos et al. (2021) provide a formal analysis showing that even bounded-rational agents experience systematic performance drops under information overload. Roetzel (2019) offers a comprehensive pre-2020 review covering causes, moderators, and consequences across domains. In digital consumer contexts, recent work (Advances in Consumer Research, 2025) documents how cognitive biases are amplified by information volume in online decision-making. McKinsey's "Bias Busters" (2026) demonstrates that cognitive overload acts as a multiplier for all other biases — it does not merely add to them but compounds their effects.

For LLM-based recommendations specifically, the problem is acute: each LLM produces verbose, well-structured responses with explanations, caveats, and follow-up questions. Across N LLMs, the user faces N x K items with detailed justifications — exactly the information volume shown to degrade decisions.

### 2.2 Choice Overload

Large option sets reduce the probability of choosing at all and reduce satisfaction with whatever is chosen. A large-scale field experiment (MSOM, 2022) established that there is an optimal recommendation list size for online purchases — beyond it, conversion rate drops. The UCLA finding ("As Few as Three Options") shows that even very small sets can trigger overload in online shopping contexts. In LLM-specific contexts, "Decisions with ChatGPT" (2023) proved that choice overload exists in AI-mediated recommendations: when ChatGPT offers larger option sets, perceived overload increases and user satisfaction decreases.

Multi-list interfaces (Examining Choice Overload, 2023) show that presenting recommendations across multiple sources (e.g., carousel layouts) increases perceived choice difficulty compared to single lists — directly relevant to multi-LLM settings.

### 2.3 Anchoring Bias

Users fixate disproportionately on the first piece of information presented. Three controlled experiments (INFORMS / Information Systems Research, 2013) demonstrate that predicted ratings from a recommender system alter users' actual evaluations of items — the anchoring effect persists even after users experience the item. Earlier work (ACM, 2010) quantified how anchoring on predicted ratings shifts users' own ratings by measurable magnitudes.

In a multi-LLM setting, the problem is compounded: each system presents its items in a different order, creating competing anchors. The user who reads GPT first is anchored differently than the user who reads Gemini first, introducing an arbitrary dependence on presentation sequence.

### 2.4 Confirmation Bias and Filter Bubbles

Each recommender reinforces its own framing of the user's query. A systematic review of filter bubbles in recommender systems (2024) finds that the interaction between user choices and system outputs creates feedback loops that progressively narrow exposure. "Deconstructing the Filter Bubble" (2020) models how user decision-making and RS outputs co-evolve into increasingly narrow recommendation spaces.

Cook and Smallman (Human Factors, 2008) provide experimental evidence that confirmation bias distorts analyst decision-making even when supported by visual evidence tools — demonstrating that decision support interfaces must actively counteract bias rather than merely present information.

In the multi-LLM context, each LLM creates its own framing bubble (e.g., one framing "food and identity" as economics, another as personal memoir). Without an intermediary, the user may gravitate toward whichever framing confirms their initial expectation.

### 2.5 Bounded Rationality and Preference Articulation Difficulty

Users cannot fully articulate what they want. "Interpretability Gone Bad" (Kaur et al., CSCW) shows that even well-designed interpretability tools can worsen decisions under bounded rationality — the tools provide correct information, but users lack the cognitive bandwidth to use it properly. Patient decision aid research (JMIR, 2016) demonstrates that explicitly helping users articulate their preferences (rather than presenting more information) improves decision quality. Randomized trials on decision aids (Fagerlin et al.) show that ordering and context effects introduce biases unless the aid is explicitly designed to counteract them.

Dialogue-based preference elicitation (ACL, 2007; Variational Reasoning, 2022) shows that multi-turn conversational interaction progressively improves preference specificity, supporting the mechanism by which a conversational Advisor can help users articulate what they cannot initially express.

---

## 3. User-Side Agents as Shields and Mediators

A growing line of work positions an LLM agent on the user's side to protect or assist users when interacting with opaque recommendation platforms.

**LLM Agent as a Shield (iAgent)** (ACL Findings 2025). Introduces a *user–agent–platform* paradigm where an LLM agent acts as a protective intermediary. The agent interprets explicit user feedback, filters advertisements, reorders recommendations, and constructs a dynamic user profile exclusively from explicit interactions — never from implicit logs. Results show improved NDCG/Recall and significantly reduced echo-chamber effects, especially for low-activity users.

**InstructAgent / Instruct2Agent** (2024). Allows users to control recommendations via free-form natural-language instructions (e.g., "more indie, fewer blockbusters"). The agent parses instructions, reorders candidates, and maintains a dynamic profile updated only by feedback and instructions. Instruct2Agent extends this with long-term memory. Achieves +16% average improvement in NDCG@10 across four datasets.

**Interactive Recommendation Feed (RecBot / IRF)** (arXiv 2025). A dual-agent system embedded in a recommendation feed: a *Parser Agent* converts natural-language commands into structured preferences, and a *Planner Agent* orchestrates tool calls and updates ranking policies in real time. Supports realistic interactive scenarios including rejections and continuations.

---

## 4. Meta-Learning and Model Selection

**MetaSelector** (2020). Rather than relying on a single recommender, MetaSelector trains a meta-learning selector that chooses the best recommender model per user from a pool of candidates. Surpasses single models and static ensembles on MovieLens and Yelp. A precedent for multi-system aggregation, though it operates at the model level rather than at the conversational level and does not interact with the user.

---

## 5. Domain-Specific Conversational Advisors

**LLM-based Personalized Portfolio Recommender (L-PPR / FinAgent)** (arXiv 2025). In the finance domain, FinAgent collects conversational input, converts subjective preferences into structured risk vectors, and conditions a reinforcement-learning policy. Demonstrates the viability of closing a conversational feedback loop in a high-stakes domain.

**Prospect Personalized Recommendation (Rec4Agentverse)** (arXiv 2024). Proposes a three-stage paradigm where *Item Agents* (domain-specialized agents) are recommended to users by an *Agent Recommender*. Introduces cross-agent collaboration as a mechanism for multi-domain advisory support.

---

## 6. Knowledge-Based and Question-Driven CRS

**Knowledge-Based Question Generation (KBQG)** (arXiv 2021). Uses a Knowledge Graph to generate personalized clarifying questions (yes/no or open-ended) that reduce conversational turns and improve recommendation precision. Models diverse KG relations alongside user embeddings.

**Q&A-Driven CRS with Multi-Interest Modeling** (Neurocomputing 2025). Integrates explicit multi-interest modeling with contextual question generation to elicit granular preferences in multi-turn Q&A dialogues. Outperforms baselines on LastFM, Yelp, and Book datasets.

**Large-Scale Interactive CRS (AC-CRS)** (UMass 2021). Unifies dialogue policy and recommendation model via reinforcement learning, learning jointly from interactive cycles that include rejections and continuations. Improves belief tracking over separated approaches.

**An Interactive Agent System for Knowledge-Based Recommendation** (ACM 2011). A pre-LLM multi-agent architecture where personal agents model user beliefs and negotiate with external knowledge/resource agents. A pioneer in agent-mediated recommendation.

---

## 7. Cognitive Bias Mitigation and Explanation Quality

**Human Factors of Confirmation Bias in Decision Support** (Cook & Smallman, Human Factors 2008). Empirical study demonstrating how confirmation bias distorts analyst decision-making even when supported by graphical evidence tools. Shows that decision support interfaces must actively counteract bias rather than merely present information.

**Cognitive Models for Self-Induced Bias** (JAISCR 2023). Agent-based simulation showing that recommenders trained on choices from cognitively limited users amplify self-induced bias over time. Proposes algorithms that adjust outputs considering cognitive limitations.

**How People Explain Action** (de Graaf & Malle, AAAI Fall Symposium 2017). Argues that when AI agents are anthropomorphized, users expect contrastive, selective, and social explanations. Provides the theoretical basis for explanation strategy: explaining "why X instead of Y" (contrastive), highlighting only decision-relevant information (selective), and framing explanations as conversational knowledge transfer (social).

**Aspect-Based Summarization for Explanations** (RecSys 2022). Generates hierarchical explanations with multiple levels of detail, adapting automatically by detected user profile. Improves comprehension +25% for novices and reduces decision time by 15–20%.

**CSLA: Explainable Recommendation via Content Summarization** (Neurocomputing 2025). Extracts key co-occurrences via complex networks and PageRank, then generates concise explanations using linear attention. State-of-the-art in explanation fidelity with reduced training cost.

**Natural Language Justifications for Recommendations** (CEUR-WS 2019). Filters and summarizes candidate justification sentences to produce concise natural-language justifications for recommended items.

---

## 8. Surveys

**A Survey on LLM-Powered Agents for Recommender Systems** (arXiv 2025). Comprehensive taxonomy of three paradigms (recommender-oriented, interaction-oriented, simulation-oriented) with four core modules (profile, memory, planning, action). Identifies critical gaps in scalability, cold-start, and offline evaluation.

---

## 9. Cross-System Consistency

**ConSCompF** (2024). A framework for evaluating output similarity between different LLMs without labeled data. Generates multiple responses per prompt, computes internal consistency via SBERT embeddings, and measures adjusted cosine similarity. Relevant as a potential tool for quantifying divergence between multiple LLM recommenders.

---

## 10. Foundations of Human-Aware AI and Hybrid System Design

**Challenges of Human-Aware AI Systems** (Kambhampati, arXiv 2019). AI systems must be explicitly designed with awareness of human mental models, expectations, and limitations. Identifies the gap between AI's narrow competence and users' tendency to overestimate its range of expertise.

**Theory of Mind in Agent-Based Systems** (de Weerd, Verbrugge & Verheij, Artificial Intelligence 2013). Second-order Theory of Mind — modeling what another agent believes about one's own beliefs — significantly improves negotiation outcomes. Relevant to the Advisor's need to maintain a model of user knowledge, beliefs, and remaining uncertainty.

**A Boxology of Design Patterns for Hybrid Learning and Reasoning Systems** (Van Harmelen & Ten Teije, Journal of Web Engineering 2019). Taxonomy of reusable architectural design patterns for systems combining learning and reasoning components. Provides a framework for describing the Advisor's architecture: LLM recommenders provide generation, the Advisor provides reasoning and synthesis.

**Interaction Design Patterns for HI Systems** (Ligthart et al., AAMAS 2019). Identifies interaction patterns for the "getting acquainted" phase of hybrid intelligent systems, with evaluation showing substantially different efficacy across patterns. Directly relevant to how the Advisor structures conversational turns.

**A Research Agenda for Hybrid Intelligence** (Akata et al., IEEE Computer 2020). Defines Hybrid Intelligence and outlines four challenges: Collaborative, Adaptive, Responsible, and Explainable. Provides the theoretical framework for the Advisor's interaction design and action space.

---

## 11. Gap Analysis

### 11.1 Gap Analysis Table

| Paper | User-Side Agent | Multi-LLM Comparison | Conversational Advisory | Info Overload Mitigation | Choice Overload Mitigation | Anchoring Mitigation | Confirmation Bias Mitigation | Preference Elicitation | Domain-Independent |
|---|---|---|---|---|---|---|---|---|---|
| iAgent (2025) | Yes | No | No | No | No | No | Partial (echo chambers) | No | No (single RS) |
| InstructAgent (2024) | Yes | No | Limited | No | No | No | No | Partial (instructions) | No |
| RecBot / IRF (2025) | Yes | No | Yes (commands) | No | No | No | No | No | No |
| MetaSelector (2020) | No | Yes (model selection) | No | No | No | No | No | No | Yes |
| FinAgent (2025) | Yes | No | Yes | No | No | No | No | Yes | No (finance) |
| Rec4Agentverse (2024) | Yes | Partial | Limited | No | No | No | No | No | Partial |
| KBQG (2021) | No | No | Yes (Qs) | No | No | No | No | Yes | No (KG-dependent) |
| Q&A CRS (2025) | No | No | Yes (Q&A) | No | No | No | No | Yes | No |
| AC-CRS (2021) | No | No | Yes (RL) | No | No | No | No | Partial | No |
| KBS Agent (2011) | Yes | Partial | Limited | No | No | No | No | Partial | No |
| Cook & Smallman (2008) | No | No | No | No | No | No | Yes | No | Yes |
| Cognitive Bias (2023) | No | No | No | Yes | No | No | No | No | Yes |
| Aspect-Based Expl. (2022) | No | No | Partial | Partial | Partial | No | No | No | Yes |
| Decisions w/ ChatGPT (2023) | No | No | No | No | Yes | No | No | No | Yes |
| INFORMS Anchoring (2013) | No | No | No | No | No | Yes | No | No | Yes |
| Filter Bubbles Review (2024) | No | No | No | No | No | No | Yes | No | Yes |
| Pothos et al. (2021) | No | No | No | Yes | No | No | No | No | Yes |
| JMIR Decision Aids (2016) | No | No | No | Partial | Partial | Partial | Partial | Yes | Yes |
| ConSCompF (2024) | No | Yes | No | No | No | No | No | No | Yes |
| Akata et al. (2020) | No | No | No | No | No | No | No | No | Yes (framework) |
| Kambhampati (2019) | No | No | No | No | No | No | No | No | Yes (framework) |
| de Weerd et al. (2013) | No | No | No | No | No | No | No | No | Yes (framework) |

### 11.2 Summary of the Gap

No existing work simultaneously addresses **all** of the following:

1. **Multi-LLM comparison from the user's side** — comparing outputs from multiple independent LLM recommenders to detect convergence, divergence, and complementary coverage.

2. **Conversational advisory interaction** — a proactive, multi-turn dialogue that helps the user navigate recommendations, not a silent pre-filter or single-shot reranker.

3. **Explicit mitigation of documented cognitive biases** — information overload, choice overload, anchoring, and confirmation bias, each addressed by specific, empirically grounded design decisions.

4. **Preference elicitation through dialogue** — helping users articulate what they cannot initially express, guided by information gain from cross-system divergence.

5. **Domain independence** — no reliance on knowledge graphs, domain databases, or curated metadata. The Advisor works across any recommendation domain.

The closest works (iAgent, InstructAgent) place an agent on the user's side but interact with a single recommender, do not engage the user conversationally, and do not ground their design in cognitive bias research. MetaSelector and ConSCompF address multi-system comparison but lack a user-facing conversational component. KBQG and Q&A CRS generate clarifying questions but within a single system, not across multiple independent recommenders. The cognitive bias literature (anchoring, choice overload, filter bubbles) documents the problems but does not propose conversational mediation as a solution. Patient decision aid research (JMIR) demonstrates that preference elicitation improves decisions but has not been applied to multi-LLM recommendation.

Our proposal fills this gap: a domain-independent conversational Advisor that mediates between the user and multiple LLM recommenders, using cross-system triangulation, contextual bandit action selection, belief state tracking, and empirically grounded debiasing strategies to help users make better decisions.

---

## References

### Cognitive Limitations and Decision Support
- Pothos, E. M., et al. (2021). Information overload for (bounded) rational agents. *Proceedings of the Royal Society B*, 288(1944).
- Roetzel, P. G. (2019). Information overload in the information age: a review of the literature. *Business Research*, 12(2), 479–522.
- Cognitive Biases in Digital Decision-Making: How Consumers Navigate Information Overload. *Advances in Consumer Research*, 2025.
- Consumer Anxiety in the Digital Age: Analyzing the Impact of Information Overload on Decision-Making. 2024.
- Bias Busters: How cognitive overload multiplies every bias. McKinsey, 2026.
- Decisions with ChatGPT: Reexamining choice overload in AI-mediated recommendations. *Journal of Retailing and Consumer Services*, 2023.
- Do Recommender Systems Manipulate Consumer Preferences? A Study of Anchoring Effects. *Information Systems Research*, INFORMS, 2013.
- Anchoring effects of recommender systems. ACM, 2010.
- Filter Bubbles in Recommender Systems: Fact or Fallacy — A Systematic Review. 2024.
- Deconstructing the Filter Bubble: User Decision-Making and Recommender Systems. 2020.
- The Choice Overload Effect in Online Recommender Systems. *Manufacturing & Service Operations Management*, 2022.
- As Few as Three Options Can Be Too Many for Online Shoppers. UCLA, 2022.
- Examining Choice Overload across Single-list and Multi-list User Interfaces. 2023.
- Kaur, H., et al. Interpretability Gone Bad: The Role of Bounded Rationality. CSCW.
- Features of Computer-Based Decision Aids: Systematic Review. *JMIR*, 2016.
- Testing Whether Decision Aids Introduce Cognitive Biases. Fagerlin et al.
- Cook, M. B. & Smallman, H. S. (2008). Human factors of the confirmation bias in intelligence analysis. *Human Factors*, 50(5), 745–754.

### User-Side Agents and CRS
- iAgent — ACL Findings 2025. https://aclanthology.org/2025.findings-acl.928/
- InstructAgent — 2024. https://www.themoonlight.io/en/review/instructagent
- RecBot / IRF — arXiv 2025. https://arxiv.org/html/2509.21317
- MetaSelector — arXiv 2020. https://arxiv.org/pdf/2001.10378.pdf
- FinAgent / L-PPR — arXiv 2025. https://www.arxiv.org/pdf/2512.12922.pdf
- Rec4Agentverse — arXiv 2024. https://arxiv.org/html/2402.18240v2
- KBQG — arXiv 2021. https://arxiv.org/pdf/2105.04774.pdf
- Q&A CRS — Neurocomputing 2025. https://doi.org/10.1016/j.neucom.2025.01.001
- AC-CRS — UMass 2021. https://people.cs.umass.edu/~pthomas/papers/Montazeralghaem2021.pdf
- KBS Agent — ACM 2011. https://doi.org/10.1145/2108616.2108681
- Dialogue Strategies for Conversational Recommender Systems. ACL, 2007.
- Variational Reasoning about User Preferences for Conversational Recommendation. 2022.

### Cognitive Bias in RS
- Cognitive Bias (Self-Induced) — JAISCR 2023. https://doi.org/10.2478/jaiscr-2023-0008

### Explanation Quality
- de Graaf, M. A. & Malle, B. F. (2017). How people explain action. *Proc. AAAI Fall Symposium*, pp. 19–26.
- Aspect-Based Expl. — RecSys 2022. https://doi.org/10.1145/3539637.3557002
- CSLA — Neurocomputing 2025. https://doi.org/10.1016/j.neucom.2025.02.001
- NL Justifications — CEUR-WS 2019. https://ceur-ws.org/Vol-2495/paper8.pdf

### Cross-System Consistency
- ConSCompF — 2024. https://paperswithcode.com/paper/conscompf

### Foundations and Frameworks
- Akata, Z., et al. (2020). A Research Agenda for Hybrid Intelligence. *IEEE Computer*, 53(8), 18–28.
- Saricks, J. G. (2005). *Readers' Advisory Service in the Public Library* (3rd ed.). ALA Editions.
- Kambhampati, S. (2019). Challenges of human-aware AI systems. arXiv:1910.07089.
- de Weerd, H., Verbrugge, R. & Verheij, B. (2013). How much does it help to know what she knows you know? *Artificial Intelligence*, 199–200, 67–92.
- Van Harmelen, F. & Ten Teije, A. (2019). A boxology of design patterns for hybrid learning and reasoning systems. *Journal of Web Engineering*, 18(1), 97–124.
- Ligthart, M., et al. (2019). A child and a robot getting acquainted: Interaction design for eliciting self-disclosure. *Proc. 18th AAMAS*, pp. 61–70.

### Surveys
- Survey on LLM-Powered Agents for RS — arXiv 2025. https://arxiv.org/html/2502.10050v1
