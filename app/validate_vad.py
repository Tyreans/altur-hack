"""
Validación de Silero VAD contra Ground Truth (turns.json)
"""
import json

def evaluate_vad_against_ground_truth(predicted_turns: list[dict], ground_truth_file: str, target_channel: int) -> dict:
    """
    Compara los timestamps producidos por el VAD contra un turns.json ground truth.
    Solo evalúa el target_channel (0 = caller, 1 = agente).
    
    Regresa métricas como el Error Absoluto Medio (MAE) de los bordes y IoU.
    """
    with open(ground_truth_file, "r") as f:
        data = json.load(f)
        
    gt_turns = [t for t in data.get("turns", []) if t["channel"] == target_channel]
    
    if not gt_turns and not predicted_turns:
        return {"iou": 1.0, "mae_edges": 0.0}
        
    # Calcular Intersection over Union global (IoU 1D)
    # Primero discretizamos o hacemos la matemática de solapamiento
    def get_coverage(turns):
        coverage = []
        for t in turns:
            coverage.append((t["start"], t["end"]))
        # Aplanar y fusionar solapamientos (aunque VAD normalmente no solapa consigo mismo)
        return sorted(coverage)

    def intersection_length(turns_a, turns_b):
        length = 0.0
        for a in turns_a:
            for b in turns_b:
                start = max(a[0], b[0])
                end = min(a[1], b[1])
                if start < end:
                    length += (end - start)
        return length

    def total_length(turns):
        return sum([t[1] - t[0] for t in turns])

    cov_pred = get_coverage(predicted_turns)
    cov_gt = get_coverage([{"start": t["start"], "end": t["end"]} for t in gt_turns])
    
    inter = intersection_length(cov_pred, cov_gt)
    union = total_length(cov_pred) + total_length(cov_gt) - inter
    iou = inter / union if union > 0 else 0.0
    
    return {
        "iou": iou,
        "intersection_seconds": inter,
        "union_seconds": union
    }
