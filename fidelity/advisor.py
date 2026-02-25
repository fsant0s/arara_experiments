import os
import ast
import json

from agents import Agent, Module, Orchestrator, User
from utils import get_llm_config

from capabilities.memories import SharedMemory
from typing import Optional, Union, List, Dict, Generator, Tuple, Any
from metrics.fidelity import pos_at_kr_ke, pos_at_kr_ke_official
from recsys.llms.user_message import user_message as user_message_template


class Advisor(Agent):
    
    DEFAULT_SYSTEM_MESSAGE = """
        You are an Advisor that helps users make better decisions about book recommendations.
        You analyze recommendations from multiple systems and help users refine their queries
        to get better, more aligned recommendations.
    """

    REFINEMENT_SUGGESTIONS = [
        "Specify a genre (e.g., mystery, romance, sci-fi)",
        "Describe the tone you want (e.g., light, dark, humorous, emotional)",
        "Mention a book you liked and want something similar to",
        "Describe your current mood or what you're looking for (e.g., escapism, learning, inspiration)",
    ]

    def __init__(
        self,
        name: Optional[str] = "advisor",
        system_message: Optional[Union[str, List]] = DEFAULT_SYSTEM_MESSAGE,
        K_r: int = 5,
        K_e: int = 3,
        overlap_threshold: int = 1,
        recsys: Optional[Agent] = None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            system_message=system_message,
            **kwargs,
        )

        self.K_r = K_r
        self.K_e = K_e
        self.overlap_threshold = overlap_threshold
        self._recsys = recsys
        self._pending_baseline: Dict[str, Dict[str, Any]] = {}
        self._query_history: List[Dict[str, Any]] = []

        self.unregister_reply_func(Agent._generate_oai_reply)
        self.register_reply(Agent, Advisor.process)

    def _parse_payload(self, content: Union[str, Dict]) -> Dict:
        """Parse payload from string or dict format."""
        if isinstance(content, dict):
            return content
        if not isinstance(content, str):
            return {}
        try:
            parsed = json.loads(content)
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                return parsed[0]
            return parsed
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(content)
                if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                    return parsed[0]
                return parsed
            except (ValueError, SyntaxError):
                return {}

    def _get_user_context(self) -> Tuple[Optional[Dict], Optional[List], Optional[List]]:
        """Get user payload, history and profile from shared memory."""
        memories = self.list_shared_memory_by_owner("user")
        if not memories:
            return None, None, None

        user_payload = memories[-1].content if hasattr(memories[-1], "content") else {}
        if isinstance(user_payload, str):
            user_payload = self._parse_payload(user_payload)
        
        history = user_payload.get("history", []) if isinstance(user_payload, dict) else []
        user_profile = [
            item.get("asin")
            for item in history
            if isinstance(item, dict) and "asin" in item
        ]
        
        return user_payload, history, user_profile

    def _extract_top_k_titles(self, payload: Dict, k: int = 3) -> Dict[str, List[str]]:
        """Extract top-K recommended titles from each RecSys."""
        recsys_rankings: Dict[str, List[str]] = {}
        
        for agent_name, agent_payload in payload.items():
            if isinstance(agent_payload, str):
                agent_payload = self._parse_payload(agent_payload)
            
            if isinstance(agent_payload, list) and agent_payload:
                agent_payload = agent_payload[0]
            
            if not isinstance(agent_payload, dict):
                continue
            
            if "error" in agent_payload:
                print(f"[Advisor] Skipping {agent_name} due to error: {agent_payload.get('error')}")
                continue
            
            recommendations = agent_payload.get("top_k_recommendations", [])
            if not recommendations:
                rec = agent_payload.get("recommendation")
                if rec and isinstance(rec, dict):
                    title = rec.get("book_title") or rec.get("title")
                    if title:
                        recommendations = [{"book_title": title}]
            
            titles = []
            for rec in recommendations[:k]:
                if isinstance(rec, dict):
                    title = rec.get("book_title") or rec.get("title")
                    if title:
                        titles.append(title.lower().strip())
            
            if titles:
                recsys_rankings[agent_name] = titles
        
        return recsys_rankings

    def _compute_overlap(self, recsys_rankings: Dict[str, List[str]]) -> Dict[str, Any]:
        """Compute overlap metrics between RecSys recommendations."""
        if len(recsys_rankings) < 2:
            return {
                "overlap_count": 0,
                "overlap_ratio": 0.0,
                "common_titles": [],
                "total_systems": len(recsys_rankings),
                "is_query_vague": True,
            }
        
        all_titles_sets = [set(titles) for titles in recsys_rankings.values()]
        
        common_titles = set.intersection(*all_titles_sets) if all_titles_sets else set()
        
        pairwise_overlaps = []
        systems = list(recsys_rankings.keys())
        for i in range(len(systems)):
            for j in range(i + 1, len(systems)):
                overlap = len(set(recsys_rankings[systems[i]]) & set(recsys_rankings[systems[j]]))
                pairwise_overlaps.append(overlap)
        
        avg_pairwise = sum(pairwise_overlaps) / len(pairwise_overlaps) if pairwise_overlaps else 0
        
        all_titles = set()
        for titles in recsys_rankings.values():
            all_titles.update(titles)
        
        overlap_ratio = len(common_titles) / len(all_titles) if all_titles else 0
        
        return {
            "overlap_count": len(common_titles),
            "overlap_ratio": overlap_ratio,
            "common_titles": list(common_titles),
            "avg_pairwise_overlap": avg_pairwise,
            "total_systems": len(recsys_rankings),
            "total_unique_titles": len(all_titles),
            "is_query_vague": len(common_titles) < self.overlap_threshold,
        }

    def _generate_refinement_message(self, overlap_metrics: Dict[str, Any], recsys_rankings: Dict[str, List[str]]) -> str:
        """Generate a message suggesting query refinement."""
        
        msg_parts = []
        msg_parts.append("=" * 50)
        msg_parts.append("ADVISOR ANALYSIS")
        msg_parts.append("=" * 50)
        msg_parts.append("")
        msg_parts.append(f"Systems analyzed: {overlap_metrics['total_systems']}")
        msg_parts.append(f"Unique recommendations: {overlap_metrics['total_unique_titles']}")
        msg_parts.append(f"Common recommendations: {overlap_metrics['overlap_count']}")
        msg_parts.append("")
        
        if overlap_metrics["is_query_vague"]:
            msg_parts.append("STATUS: Query may be too vague")
            msg_parts.append("-" * 30)
            msg_parts.append("The recommendation systems gave very different suggestions.")
            msg_parts.append("This usually means your request could be interpreted in multiple ways.")
            msg_parts.append("")
            msg_parts.append("SUGGESTIONS TO REFINE YOUR QUERY:")
            for i, suggestion in enumerate(self.REFINEMENT_SUGGESTIONS, 1):
                msg_parts.append(f"  {i}. {suggestion}")
            msg_parts.append("")
            msg_parts.append("WHAT EACH SYSTEM RECOMMENDED:")
            for agent_name, titles in recsys_rankings.items():
                msg_parts.append(f"  {agent_name}:")
                for title in titles[:3]:
                    msg_parts.append(f"    - {title.title()}")
        else:
            msg_parts.append("STATUS: Good query clarity")
            msg_parts.append("-" * 30)
            msg_parts.append("The systems show agreement on recommendations.")
            if overlap_metrics["common_titles"]:
                msg_parts.append("")
                msg_parts.append("RECOMMENDED (consensus):")
                for title in overlap_metrics["common_titles"]:
                    msg_parts.append(f"  - {title.title()}")
        
        msg_parts.append("")
        msg_parts.append("=" * 50)
        
        return "\n".join(msg_parts)

    def _generate_consolidated_response(
        self, 
        payload: Dict, 
        overlap_metrics: Dict[str, Any],
        recsys_rankings: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """Generate consolidated response with recommendations and analysis."""
        
        response = {
            "query_analysis": {
                "is_vague": overlap_metrics["is_query_vague"],
                "overlap_count": overlap_metrics["overlap_count"],
                "overlap_ratio": overlap_metrics["overlap_ratio"],
                "total_systems": overlap_metrics["total_systems"],
            },
            "recommendations_by_system": {},
            "consensus_recommendations": overlap_metrics["common_titles"],
        }
        
        for agent_name, agent_payload in payload.items():
            if isinstance(agent_payload, str):
                agent_payload = self._parse_payload(agent_payload)
            if not isinstance(agent_payload, dict):
                continue
            
            recommendations = agent_payload.get("top_k_recommendations", [])
            response["recommendations_by_system"][agent_name] = recommendations
        
        if overlap_metrics["is_query_vague"]:
            response["advisor_message"] = self._generate_refinement_message(overlap_metrics, recsys_rankings)
            response["action_required"] = "refine_query"
            response["suggestions"] = self.REFINEMENT_SUGGESTIONS
        else:
            response["advisor_message"] = self._generate_refinement_message(overlap_metrics, recsys_rankings)
            response["action_required"] = None
        
        return response

    def process(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Agent] = None,
        config: Optional[Module] = None,
    ) -> Generator[Tuple[bool, Optional[str]], None, None]:
        """Main routing: dispatch to appropriate handler based on sender."""
        
        if messages is None or not messages:
            yield [(True, json.dumps({"error": "No messages received."}))]
            return
        if sender is None:
            yield [(True, json.dumps({"error": "Sender not provided."}))]
            return

        print(f"[Advisor] Received message from: {sender.name}")

        if isinstance(sender, User) or sender.name == "user":
            yield from self._handle_user_message(messages, sender)
        elif sender.name == "recsys_orchestrator" or isinstance(sender, Orchestrator):
            yield from self._handle_recsys_response(messages, sender)
        else:
            yield [(True, json.dumps({"error": f"Unknown sender: {sender.name}"}))]

    def _handle_user_message(
        self,
        messages: List[Dict],
        sender: Agent,
    ) -> Generator[Tuple[bool, Optional[str]], None, None]:
        """
        Handle messages from User (advisory mode).
        User can:
        - Send baseline payload directly for diagnostic
        - Update preferences (delta P_u)
        - Ask questions about recommendations
        """
        print("[Advisor] Processing user message (advisory mode)")
        
        payload = self._parse_payload(messages[-1].get("content"))
        
        if not payload or not isinstance(payload, dict):
            yield [(True, json.dumps({
                "error": "Invalid payload format.",
                "hint": "Send a JSON with recommendation payloads or preference updates."
            }))]
            return

        user_payload, history, user_profile = self._get_user_context()
        if not user_payload:
            yield [(True, json.dumps({"error": "No shared memory for owner 'user'."}))]
            return
        if not user_profile:
            yield [(True, json.dumps({"error": "User profile is empty or missing."}))]
            return

        is_recsys_payload = any(
            isinstance(v, dict) and ("recommendation" in v or "baseline_ranking" in v)
            for v in payload.values()
        )

        if is_recsys_payload:
            yield from self._process_baseline_payload(payload, user_payload, history, user_profile, sender)
        else:
            yield [(True, json.dumps({
                "message": "Advisory mode: received user input.",
                "payload_keys": list(payload.keys()),
                "hint": "Send recommendation payloads to compute fidelity metrics."
            }))]

    def _handle_recsys_response(
        self,
        messages: List[Dict],
        sender: Agent,
    ) -> Generator[Tuple[bool, Optional[str]], None, None]:
        """
        Handle messages from RecSys (diagnostic mode).
        RecSys sends recommendation payloads -> Advisor analyzes and responds.
        
        Flow:
        1. Extract top-K recommendations from each RecSys
        2. Compute overlap between systems
        3. If overlap < threshold: suggest query refinement
        4. If overlap >= threshold: present consolidated recommendations
        """
        print("[Advisor] Processing recsys response (diagnostic mode)")
        payload = self._parse_payload(messages[-1].get("content"))
        
        if not payload or not isinstance(payload, dict):
            yield [(True, json.dumps({"error": "Payload must be a dict mapping agent names to outputs."}))]
            return

        user_payload, history, user_profile = self._get_user_context()
        if not user_payload:
            yield [(True, json.dumps({"error": "No shared memory for owner 'user'."}))]
            return

        recsys_rankings = self._extract_top_k_titles(payload, k=self.K_r)
        
        if not recsys_rankings:
            yield [(True, json.dumps({"error": "Could not extract recommendations from RecSys payloads."}))]
            return
        
        overlap_metrics = self._compute_overlap(recsys_rankings)
        
        self._query_history.append({
            "query": user_payload.get("instruction", ""),
            "overlap_metrics": overlap_metrics,
            "recsys_rankings": recsys_rankings,
        })
        
        print(f"[Advisor] Overlap analysis: {overlap_metrics['overlap_count']} common items, "
              f"query_vague={overlap_metrics['is_query_vague']}")
        
        response = self._generate_consolidated_response(payload, overlap_metrics, recsys_rankings)
        
        print(response["advisor_message"])
        
        yield [(True, json.dumps(response, indent=2))]

    def _process_baseline_payload(
        self,
        payload: Dict,
        user_payload: Dict,
        history: List,
        user_profile: List,
        sender: Agent,
    ) -> Generator[Tuple[bool, Optional[str]], None, None]:
        """Process baseline payload and request counterfactual from RecSys."""
        
        K_e = self.K_e
        baseline_payload: Dict[str, Dict] = {}
        removal_items: List[Any] = []

        for agent_name, agent_payload in payload.items():
            if isinstance(agent_payload, list):
                agent_payload = agent_payload[0] if agent_payload else {}
            if not isinstance(agent_payload, dict):
                continue

            explanation_items_ordered = agent_payload.get("explanation_items_ordered", [])
            baseline_ranking = agent_payload.get("baseline_ranking", [])
            recommendation = agent_payload.get("recommendation")

            if not explanation_items_ordered or not baseline_ranking or not recommendation:
                continue

            baseline_payload[agent_name] = agent_payload

            for item_id in explanation_items_ordered[:K_e]:
                if item_id not in removal_items:
                    removal_items.append(item_id)

        if not baseline_payload:
            yield [(True, json.dumps({"error": "No valid baseline payloads found."}))]
            return

        removal_items = removal_items[:K_e]
        if not removal_items:
            yield [(True, json.dumps({"error": "No explanation items found for counterfactual."}))]
            return

        filtered_history = [
            item for item in history
            if isinstance(item, dict) and item.get("asin") not in removal_items
        ]
        filtered_profile = [
            item.get("asin") for item in filtered_history
            if isinstance(item, dict) and "asin" in item
        ]

        read_books = ""
        for item in filtered_history:
            read_books += (
                f"- ID: {item.get('asin')}\n"
                f"  Name: {item.get('title')}\n"
                f"  Description: {item.get('description')}\n"
                f"  User Review: {item.get('review')}\n\n"
            )

        counterfactual_message = user_message_template.format(
            persona=user_payload.get("persona", ""),
            read_books=read_books,
            instruction=user_payload.get("instruction", ""),
        )

        recsys = self._recsys or sender
        recsys_key = recsys.name if hasattr(recsys, 'name') else sender.name
        
        self._pending_baseline[recsys_key] = {
            "baseline_payload": baseline_payload,
            "removal_items": removal_items,
            "filtered_profile": filtered_profile,
        }
        
        print(f"[Advisor] Requesting counterfactual from {recsys_key}")
        self.send(
            message=counterfactual_message,
            recipient=recsys,
            request_reply=True,
            silent=False,
        )

        yield [(True, None)]

    def _compute_counterfactual_metrics(
        self,
        cf_payload: Dict,
        sender: Agent,
        user_profile: List,
    ) -> Generator[Tuple[bool, Optional[str]], None, None]:
        """Compute fidelity metrics comparing baseline vs counterfactual."""
        
        print("[Advisor] Computing counterfactual metrics")
        
        baseline_state = self._pending_baseline.pop(sender.name)
        baseline_payload = baseline_state["baseline_payload"]
        removal_items = baseline_state["removal_items"]
        filtered_profile = baseline_state["filtered_profile"]

        K_r = self.K_r
        K_e = self.K_e
        results: Dict[str, Dict] = {}
        errors: Dict[str, Dict] = {}

        for agent_name, agent_payload in baseline_payload.items():
            if not isinstance(agent_payload, dict):
                errors[agent_name] = {"error": "Invalid baseline payload for agent."}
                continue

            cf_agent_payload = cf_payload.get(agent_name)
            if not isinstance(cf_agent_payload, dict):
                errors[agent_name] = {"error": "Missing counterfactual payload for agent."}
                continue

            baseline_ranking = agent_payload.get("baseline_ranking", [])
            recommendation = agent_payload.get("recommendation") or {}
            target_item = recommendation.get("item_id") or recommendation.get("asin")
            counterfactual_ranking = cf_agent_payload.get("baseline_ranking", [])
            counterfactual_recommendation = cf_agent_payload.get("recommendation")

            if not baseline_ranking or not counterfactual_ranking or target_item is None:
                errors[agent_name] = {
                    "error": "Missing required baseline or counterfactual fields.",
                    "required": ["baseline_ranking", "counterfactual baseline_ranking", "recommendation.item_id"],
                }
                continue

            baseline_key = tuple(user_profile)
            cf_key = tuple(filtered_profile)

            def make_rank_fn(bl_ranking, cf_ranking, bl_key, c_key):
                def rank_fn(profile: List) -> List:
                    key = tuple(profile)
                    if key == bl_key:
                        return bl_ranking
                    if key == c_key:
                        return cf_ranking
                    return []
                return rank_fn

            rank_fn = make_rank_fn(baseline_ranking, counterfactual_ranking, baseline_key, cf_key)

            K_e_effective = min(K_e, len(removal_items))
            pos_value = pos_at_kr_ke(
                rank_fn=rank_fn,
                user_profile=user_profile,
                explanation_items_ordered=removal_items,
                K_r=K_r,
                K_e=K_e_effective,
            )
            pos_official = pos_at_kr_ke_official(
                rank_fn=rank_fn,
                user_profile=user_profile,
                target_item=target_item,
                explanation_items_ordered=removal_items,
                K_r=K_r,
                K_e=K_e_effective,
            )

            results[agent_name] = {
                "metrics": {
                    "pos_at_kr_ke": pos_value,
                    "pos_at_kr_ke_official": pos_official,
                },
                "K_r": K_r,
                "K_e": K_e_effective,
                "target_item": target_item,
                "baseline_ranking": baseline_ranking,
                "counterfactual_ranking": counterfactual_ranking,
                "baseline_recommendation": recommendation,
                "counterfactual_recommendation": counterfactual_recommendation,
                "removed_items": removal_items,
            }

        response = {"results": results}
        if errors:
            response["errors"] = errors

        yield [(True, json.dumps(response))]
        

def create_advisor(
    shared_memory: SharedMemory, 
    recsys: Optional[Agent] = None,
    overlap_threshold: int = 1,
) -> Advisor:
    """
    Create an Advisor agent.
    
    Args:
        shared_memory: SharedMemory skill for accessing user context
        recsys: Optional RecSys orchestrator for counterfactual requests
        overlap_threshold: Minimum number of common recommendations to consider query as "clear"
                          (default=1, meaning at least 1 common item across all systems)
    """
    advisor = Advisor(
        llm_config = get_llm_config(
            client="openrouter",
            model="openai/gpt-4o",
            api_key=os.getenv("OPEN_ROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
        ),
        system_message=Advisor.DEFAULT_SYSTEM_MESSAGE,
        skills=[shared_memory],
        recsys=recsys,
        overlap_threshold=overlap_threshold,
    )

    recomendation_module = Module(
        name="recomendation_module",
        agents=[advisor],
    )

    recommendation_orchestrator = Orchestrator(
        name="recommendation_orchestrator",
        module=recomendation_module,
        description="Orchestrator for recommendation module.",
    )

    return advisor# recommendation_orchestrator

