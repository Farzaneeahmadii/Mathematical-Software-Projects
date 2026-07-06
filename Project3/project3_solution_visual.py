# -*- coding: utf-8 -*-
"""
 پروژه سوم - نرم افزارهای ریاضی
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.interpolate import CubicSpline
except ImportError:
    CubicSpline = None


def load_runtime_data(excel_path):
    df = pd.read_excel(excel_path)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns={df.columns[0]: "Data size"})
    df["KB"] = df["Data size"].astype(str).str.upper().str.replace("KB", "", regex=False).str.strip().astype(int)
    return df.sort_values("KB").reset_index(drop=True)


def add_extra_size(df):
    output = df.copy()
    if 700 not in set(output["KB"]):
        output = pd.concat([
            output,
            pd.DataFrame([{"Data size":"700KB", "Alg.1":80, "Alg.2":320, "Alg.3":700, "KB":700}])
        ], ignore_index=True)
    return output.sort_values("KB").reset_index(drop=True)


def make_runtime_charts(df, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    algorithms = [c for c in df.columns if c.startswith("Alg")]
    labels = df["Data size"].tolist()

    # 1) نمودار میله‌ای افقی گروهی
    y = np.arange(len(labels))
    h = 0.23
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    color_list = ["#6a4c93", "#1982c4", "#8ac926"]
    hatch_list = ["//", "..", "xx"]
    for i, col in enumerate(["Alg.3", "Alg.2", "Alg.1"]):
        bars = ax.barh(y + (i - 1) * h, df[col], h, label=col,
                       color=color_list[i], hatch=hatch_list[i], edgecolor="black", linewidth=0.5)
        ax.bar_label(bars, padding=3, fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Running time")
    ax.set_ylabel("Data size")
    ax.set_title("Horizontal bar chart of algorithm runtimes")
    ax.legend(title="Algorithm", loc="lower right")
    ax.grid(axis="x", linestyle=":", alpha=0.45)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_dir / "bar_chart_different.png", dpi=220)
    plt.close(fig)

    # 2) نمودار خطی 
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    style_map = {
        "Alg.1": {"marker":"D", "linestyle":"--", "color":"#ff595e"},
        "Alg.2": {"marker":"s", "linestyle":"-.", "color":"#ffca3a"},
        "Alg.3": {"marker":"^", "linestyle":"-", "color":"#1982c4"},
    }
    for col in algorithms:
        ax.plot(labels, df[col], linewidth=2.4, markersize=7, label=col, **style_map[col])
    ax.fill_between(labels, df["Alg.1"], df["Alg.3"], alpha=0.08, color="#1982c4")
    ax.set_title("Runtime trend after adding 700KB row")
    ax.set_xlabel("Input size")
    ax.set_ylabel("Runtime")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper left", ncol=3)
    fig.tight_layout()
    fig.savefig(out_dir / "line_chart_different.png", dpi=220)
    plt.close(fig)

    # 3) نمودار جعبه‌ای افقی و notched
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    bp = ax.boxplot([df[c] for c in algorithms], labels=algorithms, vert=False,
                    notch=True, patch_artist=True, showmeans=True,
                    meanprops={"marker":"*", "markerfacecolor":"black", "markeredgecolor":"black", "markersize":8})
    for patch, color in zip(bp["boxes"], ["#ffd6a5", "#caffbf", "#9bf6ff"]):
        patch.set_facecolor(color)
        patch.set_edgecolor("#333333")
    for median in bp["medians"]:
        median.set_color("#d00000")
        median.set_linewidth(2)
    ax.set_title("Horizontal box plot of runtime distribution")
    ax.set_xlabel("Runtime")
    ax.grid(axis="x", linestyle=":", alpha=0.45)
    fig.tight_layout()
    fig.savefig(out_dir / "box_plot_different.png", dpi=220)
    plt.close(fig)


def lagrange_polynomial(x, y):
    result = np.poly1d([0.0])
    for i in range(len(x)):
        term = np.poly1d([1.0])
        denominator = 1.0
        for j in range(len(x)):
            if i != j:
                term *= np.poly1d([1.0, -x[j]])
                denominator *= x[i] - x[j]
        result += y[i] * term / denominator
    return result


def spline_report(x, y):
    if CubicSpline is None:
        return "scipy نصب نیست؛ برای چاپ ضرایب اسپلاین باید scipy نصب شود."
    spl = CubicSpline(x, y, bc_type="natural")
    rows = []
    for i in range(len(x)-1):
        a, b, c, d = spl.c[:, i]
        rows.append((x[i], x[i+1], a, b, c, d))
    return rows


def f(x):
    return np.exp(x**2)


def trapezoidal(a, b, n):
    xs = np.linspace(a, b, n + 1)
    h = (b - a) / n
    return h * (f(xs[0]) / 2 + f(xs[1:-1]).sum() + f(xs[-1]) / 2)


def simpson(a, b, n):
    if n % 2 == 1:
        n += 1
    xs = np.linspace(a, b, n + 1)
    h = (b - a) / n
    return h / 3 * (f(xs[0]) + f(xs[-1]) + 4 * f(xs[1:-1:2]).sum() + 2 * f(xs[2:-1:2]).sum())


def gaussian_quadrature(a, b, nodes_count):
    nodes, weights = np.polynomial.legendre.leggauss(nodes_count)
    mapped = (b - a) * nodes / 2 + (a + b) / 2
    return (b - a) / 2 * np.sum(weights * f(mapped))


def main():
    out = Path("outputs_second_visual")
    out.mkdir(exist_ok=True)
    excel_name = "پروژه 3.xlsx"

    points_x = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    points_y = np.array([1, 3, 5, 8, 5, 2], dtype=float)
    print("Lagrange polynomial:")
    print(lagrange_polynomial(points_x, points_y))
    print("\nNatural spline pieces:")
    print(spline_report(points_x, points_y))

    data = load_runtime_data(excel_name)
    avg_alg2 = data.loc[data["KB"].between(100, 600), "Alg.2"].mean()
    data_after = add_extra_size(data)
    data_after.drop(columns="KB").to_excel(out / "excel_after_700KB.xlsx", index=False)
    make_runtime_charts(data_after, out)
    print("\nAverage of Alg.2 for 100KB to 600KB:", avg_alg2)

    print("\nIntegral of exp(x^2) on [0,1]")
    print("Trapezoidal:", trapezoidal(0, 1, 1000))
    print("Simpson:", simpson(0, 1, 1000))
    print("Gaussian:", gaussian_quadrature(0, 1, 10))


if __name__ == "__main__":
    main()
