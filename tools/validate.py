from data.loader import DATASET, get_columns, get_stats

VALID_METRICS = set(get_columns()) - {"company", "year"}

def check_metrics(metrics: list) -> dict:
    missing = [m for m in metrics if m not in VALID_METRICS]
    return {
        "valid": len(missing) == 0,
        "missing": missing,
        "available": sorted(VALID_METRICS),
    }

def check_threshold(metric: str, op: str, value: float) -> dict:
    if metric not in VALID_METRICS:
        return {"valid": False, "reason": f"{metric} not in dataset"}
    stats = get_stats(metric)
    # Warn if threshold would likely return 0 results
    if op in (">", ">="):
        feasible = value < stats["max"]
        suggestion = round(stats["mean"], 1)
    else:
        feasible = value > stats["min"]
        suggestion = round(stats["mean"], 1)
    return {
        "valid": True,
        "feasible": feasible,
        "stats": stats,
        "suggestion": suggestion,
        "warning": None if feasible else f"Threshold {op}{value} may return 0 results (max={stats['max']})"
    }

def validate_query(q: dict) -> tuple[bool, str]:
    metrics = q.get("metrics", [])
    result = check_metrics(metrics)
    if not result["valid"]:
        return False, f"Unknown metrics: {result['missing']}. Available: {result['available']}"
    for cond in q.get("conditions", []):
        t = check_threshold(cond["metric"], cond["op"], cond["value"])
        if not t["valid"]:
            return False, t["reason"]
    return True, ""
