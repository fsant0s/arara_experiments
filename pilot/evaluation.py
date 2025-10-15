
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
    Avalia a resposta do Arara e, opcionalmente, mostra os resultados de um modelo existente.

    Args:
        dataloader: instância de Dataloader já configurada para o dataset em uso.
        data: item do dataset contendo pelo menos 'data_idx' e 'movieSubset'.
        arara_response: resposta produzida pelo agente (itens separados por [SEP]).
        model_name: se fornecido, imprime também a predição e métricas pré-calculadas desse modelo.
        k: top-k para avaliar (0 = todas as recomendações da resposta).

    Returns:
        Um dicionário com as métricas do Arara (tempo real) e, se disponível, do modelo.
    """

    results: Dict[str, Any] = {}

    # ===== Avaliação do Arara (tempo real) =====
    arara_eval = dataloader.evaluate_response(arara_response, data_idx=data["data_idx"], k=k)
    results["arara_eval"] = arara_eval

    # ===== Depois: resultados do ARARA =====
    print("=" * 80)
    print("=== ARARA ===")
    print(f"FTR:        {arara_eval['ftr']}")
    print(f"Predicted:  {arara_response}")
    print(f"Matches:    {", ".join(arara_eval['matched_titles'])}")
    print("-- Metrics --")
    print(f"Recall:    {_fmt_num(arara_eval['recall'])}")
    print(f"Precision: {_fmt_num(arara_eval['precision'])}")
    print(f"NDCG:      {_fmt_num(arara_eval['ndcg'])}")
    # print(f"Satisfied Ratio: {_fmt_num(arara_eval['satisfied_ratio'])}")

     # ===== Modelo existente primeiro (se disponível) =====
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
            print(f"Recall:    {_fmt_num(eval_result.get('recall'))}")  #how many of the relevant movies were successfully recommended — it reflects coverage of the recommendations
            print(f"Precision: {_fmt_num(eval_result.get('precision'))}") #how many of the recommended movies are actually relevant — it reflects accuracy of the recommendations.
            print(f"NDCG:      {_fmt_num(eval_result.get('ndcg'))}") #how well the order of the recommended movies matches the ideal (ground truth) order.
            #print(f"Satisfied Ratio: {_fmt_num(eval_result.get('satisfied_ratio'))}")
        else:
            print("Sem métricas pré-calculadas.")
   
    # ===== Contexto (ground truth) =====
    print("=" * 80)
    print("=== Ground truth ===")
    print(f"Itens:   {arara_eval['ground_truth']}")

    return results

__all__ = ["report_metrics"]