"""Helpers for Jupyter/IPython notebooks."""
from typing import Dict

import numpy as np
import torch
import IPython.core.display as ipython

from .run import tensor_to


def print_html(s: str) -> None:
    """Display given HTML in the IPython output cell."""
    ipython.display(ipython.HTML(s))


def summarize(t: torch.Tensor, name: str = "", prec=2, tiny=1e-6) -> None:
    """Print a small table giving statistics (mean, #NaNs, etc.) of a given tensor."""
    # Compute statistics.
    with torch.no_grad():
        N = t.numel()
        t_flat = torch.flatten(t).to(dtype=torch.float32)
        std, mean = torch.std_mean(t)

        stats: Dict[str, float] = {
            "min": torch.min(t).item(),
            "max": torch.max(t).item(),
            "mean": mean.item(),
            "std": std.item(),
            # "median": torch.median(t).item(),
            "n0": torch.linalg.norm(t_flat, ord=0).item() / N,  # proportion of non-zeroes
            "n1": torch.linalg.norm(t_flat, ord=1).item() / N,  # mean of abs
            "n2": torch.linalg.norm(t_flat, ord=2).item() / np.sqrt(N),  # quadratic mean
            # "ninfty": torch.linalg.norm(t_flat, ord=np.inf).item(),  # max of abs
            "n_nontiny": (t.abs() > tiny).sum().item() / N,  # proportion of non-tiny entries
            "n_nan": t.isnan().sum().item() / N  # proportion of NaN entries
        }

        percentiles = [0, 10, 20, 50, 80, 90, 100]
        perc_tensor = torch.Tensor([p / 100 for p in percentiles])
        quantiles = torch.quantile(t_flat,  tensor_to(perc_tensor, t_flat))

    # Format numbers with given precision (except quantiles).
    s = {key: f"{value:.{prec}g}" for key, value in stats.items()}
    for key in ["n0", "n_nontiny", "n_nan"]:
        s[key] = f"{stats[key] * 100:.{prec}f}"

    # Arrange the resulting strings.
    shape = "×".join(str(i) for i in t.shape)
    result = {
        name: f"<div style='min-width: 5em'>{shape}</div>",
        "min..max": f"{s['min']} ..{s['max']}",
        "mean±std": f"{s['mean']} ± {s['std']}",
        "mean(abs)": s["n1"],
        "2-mean": s["n2"],
        "#NaN": s["n_nan"] + "%",
        "#non-0": s["n0"] + "%",
        "#non-tiny": s["n_nontiny"] + "%",
        "quantiles": ""
    }
    if stats["n_nan"] > 0:
        result["#NaN"] = f"<span style='color: red; font-weight: bold'>{result['#NaN']}</span>"
    else:
        result["#NaN"] = '–'

    for p, q in zip(percentiles, quantiles):
        result[str(p)] = f"{q:.1g}"

    # Format `result` as an HTML table and print it.
    html_keys = "".join(f"<th>{key}</th>" for key in result.keys())
    html_values = "".join(f"<td>{val}</td>" for val in result.values())
    html = f"<thead><tr>{html_keys}</tr></thead><tbody><tr>{html_values}</tr></tbody>"
    html = f"<table>{html}</table>"
    print_html(html)


def widen_outputs():
    """Allow pictures in cell outputs to overflow to the right.

    We also swap the default from scaled to unconfined view: click an image
    to make it scaled again.
    """
    print_html("""<style>
        .output_subarea, .output_subarea img {
            overflow-x: visible !important;
            background: white !important;
            max-width: 100vw !important;
        }
        .output_subarea img.unconfined {
            max-width: 100% !important;
        }
    </style>""")
