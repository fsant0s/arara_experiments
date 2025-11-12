from typing import Any, Dict, Optional


def _fmt_num(val: Any) -> str:
    """Helper to format floats with 3 decimals; fallback to 'N/A'."""
    try:
        return f"{float(val):.3f}"
    except Exception:
        return "N/A"


def report_metrics(
    *,
    dataloader,
    data: Dict[str, Any],
    arara_response: str,
    model_name: Optional[str] = "llama-3.1-70b-instruct",
    k: int = 0,
) -> Dict[str, Any]:
    """
    Evaluates the Arara response and optionally shows the results of an existing model.

    Args:
        dataloader: instance of Dataloader already configured for the dataset in use.
        data: dataset item containing at least 'data_idx' and 'movieSubset'.
        arara_response: response produced by the agent (items separated by [SEP]).
        model_name: if provided, also prints the prediction and pre-calculated metrics of that model.
        k: top-k to evaluate (0 = all recommendations from the response).

    Returns:
        A dictionary with Arara's real-time metrics and, if available, the model’s metrics.
    """

    results: Dict[str, Any] = {}

    # ===== Arara evaluation (real-time) =====
    arara_eval = dataloader.evaluate_response(arara_response, data_idx=data["data_idx"], k=k)
    results["arara_eval"] = arara_eval

    # ===== Then: ARARA results =====
    print("=" * 80)
    print("=== ARARA ===")
    print(f"FTR:        {arara_eval['ftr']}")
    print(f"Predicted:  {arara_response}")
    print(f"Matches:    {', '.join(arara_eval['matched_titles'])}")
    print("-- Metrics --")
    print(f"Recall:    {_fmt_num(arara_eval['recall'])}")
    print(f"Precision: {_fmt_num(arara_eval['precision'])}")
    print(f"NDCG:      {_fmt_num(arara_eval['ndcg'])}")
    # print(f"Satisfied Ratio: {_fmt_num(arara_eval['satisfied_ratio'])}")

    # ===== Existing model first (if available) =====
    if model_name:
        prediction = dataloader.get_result(data_idx=data["data_idx"], model_name=model_name)
        eval_result = dataloader.get_eval_result(data_idx=data["data_idx"], model_name=model_name)
        results["model_eval"] = eval_result

        print("=" * 80)
        print(f"=== Used model: {model_name} ===")
        if prediction:
            print(f"Predicted: {prediction.get('response', '')}")
        else:
            print("Predicted: N/A")

        print("-- Metrics --")
        if eval_result:
            print(f"Recall:    {_fmt_num(eval_result.get('recall'))}")  # how many relevant movies were successfully recommended — reflects coverage
            print(f"Precision: {_fmt_num(eval_result.get('precision'))}")  # how many recommended movies are actually relevant — reflects accuracy
            print(f"NDCG:      {_fmt_num(eval_result.get('ndcg'))}")  # how well the recommendation order matches the ideal (ground truth) order
            # print(f"Satisfied Ratio: {_fmt_num(eval_result.get('satisfied_ratio'))}")
        else:
            print("No pre-calculated metrics available.")
   
    # ===== Context (ground truth) =====
    print("=" * 80)
    print("=== Ground truth ===")
    print(f"Items:   {arara_eval['ground_truth']}")

    return results

__all__ = ["report_metrics"]
