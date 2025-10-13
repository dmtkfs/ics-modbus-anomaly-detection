def fig_name(team: str, model: str, metric: str) -> str:
    safe_model = model.replace(" ", "").lower()
    safe_metric = metric.replace(" ", "").lower()
    return f"figures/{team}/{safe_model}_{safe_metric}.png"
