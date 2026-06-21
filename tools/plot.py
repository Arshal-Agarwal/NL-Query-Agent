import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import io
from datetime import datetime
from PIL import Image
from utils.normalize import normalize, should_normalize

_OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")

def render_plot(plot_decision: dict, result: list, trend_data: list,
                x_metric: str, y_metric: str, intent: list, title: str = "Results",
                x_metric_scatter: str = None) -> tuple:
    """Returns (pil_image, saved_path)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_type = plot_decision.get("plot_type", "bar")
    do_norm = should_normalize(intent, y_metric)

    if plot_type == "line" and trend_data:
        companies = [k for k in trend_data[0].keys() if k != "year"]
        years = [row["year"] for row in trend_data]
        for company in companies:
            vals = [row.get(company, 0) for row in trend_data]
            if do_norm:
                vals = normalize(vals)
            ax.plot(years, vals, marker="o", label=company)
        ax.set_xlabel("Year")
        ax.set_ylabel(f"{y_metric} (normalized)" if do_norm else y_metric)
        ax.legend(fontsize=7)

    elif plot_type == "scatter" and result and x_metric_scatter:
        xs = [r.get(x_metric_scatter, 0) for r in result]
        ys = [r.get(y_metric, 0) for r in result]
        labels = [r.get("company", "") for r in result]
        ax.scatter(xs, ys, alpha=0.7)
        for label, x, y in zip(labels, xs, ys):
            ax.annotate(label, (x, y), fontsize=6, alpha=0.7,
                        xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel(x_metric_scatter)
        ax.set_ylabel(y_metric)

    else:
        companies = [r["company"] for r in result]
        vals = [r.get(y_metric, 0) for r in result]
        if do_norm:
            vals = normalize(vals)
        ax.bar(companies, vals)
        ax.set_xlabel("Company")
        ax.set_ylabel(f"{y_metric} (normalized)" if do_norm else y_metric)
        plt.xticks(rotation=30, ha="right")

    ax.set_title(title)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    pil_img = Image.open(buf).copy()
    os.makedirs(_OUTPUTS_DIR, exist_ok=True)
    path = os.path.join(_OUTPUTS_DIR, f"plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    pil_img.save(path)
    plt.close()
    return pil_img, path
