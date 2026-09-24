import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt



HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(HERE, "..", "txt", "results.txt")  # raw experiment log from main.py
FIGDIR = os.path.join(HERE, "..", "figures")

METHODS = ["LSH", "PQ", "HNSW"]
COLOURS = {"LSH": "#2a78d6", "PQ": "#eb6834", "HNSW": "#1baf7a"}  # validated colourblind-safe palette, fixed order
DATASETS = ["random", "clustered", "financebench"]
DATASET_LABEL = {"random": "random", "clustered": "clustered", "financebench": "FinanceBench"}

INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#dcdcd8"
FLAT = "#8a8a84"



def parse_results(path=RESULTS_PATH):
    """
    Parse the main.py output log into one record per experiment run

    :param path: path to the results log
    :return: list of runs, each {dataset, metric, method, flat, rows}
    """
    runs, cur = [], None
    pending_build, shared_build = None, None

    for line in open(path):
        line = line.strip()

        header = re.match(r"=+ (.+?) =+$", line)
        if header:
            title = header.group(1)
            if "Flat" in title:  # every run starts with its exact search baseline
                dataset = re.search(r"\((\w+)", title).group(1)
                metric = "cosine" if "Cosine" in title else "l2"
                cur = {"dataset": dataset, "metric": metric, "method": None, "flat": None, "rows": []}
                runs.append(cur)
            cur["title"] = title
            continue

        build = re.match(r"build time: ([\d.]+)", line)
        if build:
            pending_build = float(build.group(1))
            continue

        result = re.match(r"recall@10: ([\d.]+).*latency: ([\d.]+)s \| index size: ([\d.]+)MB", line)
        if not result:
            continue

        recall, latency, size = map(float, result.groups())
        title = cur["title"]

        if "Flat" in title:
            cur["flat"] = {"build": pending_build, "latency": latency, "size": size}
        else:
            method = title.split()[0]
            cur["method"] = method

            if method == "HNSW" and pending_build is None:
                build_time = shared_build  # efSearch rows reuse the M=32 index, no rebuild
            else:
                build_time = pending_build
            if method == "HNSW" and "M=32" in title and pending_build is not None:
                shared_build = pending_build

            params = dict((k, int(v)) for k, v in re.findall(r"(\w+)=(\d+)", title))
            if method == "LSH":
                params["nbits"] = int(re.search(r"(\d+) bits", title).group(1))

            cur["rows"].append({"params": params, "recall": recall, "build": build_time,
                                "latency": latency, "size": size,
                                "speedup": cur["flat"]["latency"] / latency})  # same-run Flat, so nb and nq cancel out

        pending_build = None

    return runs



def get_run(runs, method, dataset, metric):
    """
    Find one run by method, dataset and metric. If the log holds the same run
    twice, the later one wins, so a rerun can simply be appended to results.txt
    """
    for r in reversed(runs):
        if r["method"] == method and r["dataset"] == dataset and r["metric"] == metric:
            return r
    return None



def frontier(rows, key, higher_is_better):
    """
    Keep only configurations not beaten on both cost and recall, sorted by cost

    :param rows: result rows
    :param key: cost column, e.g. "size", "build", "speedup"
    :param higher_is_better: True for speedup, False for size and build time
    :return: frontier rows
    """
    def better_or_equal(a, b):
        return a >= b if higher_is_better else a <= b

    keep = []
    for p in rows:
        dominated = any(
            q is not p and better_or_equal(q[key], p[key]) and q["recall"] >= p["recall"]
            and (q[key] != p[key] or q["recall"] > p["recall"])
            for q in rows
        )
        if not dominated:
            keep.append(p)

    return sorted(keep, key=lambda r: r[key])



def style_axes(ax, xlabel, ylabel, title, logx=False):
    """
    Apply a consistent, recessive axis style
    """
    if logx:
        ax.set_xscale("log")

    ax.set_xlabel(xlabel, fontsize=9, color=INK_MUTED)
    ax.set_ylabel(ylabel, fontsize=9, color=INK_MUTED)
    ax.set_title(title, fontsize=10, color=INK, pad=6)
    ax.set_ylim(0, 1.05)

    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(labelsize=8, colors=INK_MUTED, length=3)



def save(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    out = os.path.join(FIGDIR, name)
    fig.savefig(out, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)



def plot_parameter_sweeps(runs, dataset="clustered", metric="cosine"):
    """
    Report section 2: recall against each method's own parameter, one dataset and metric
    """
    lsh = get_run(runs, "LSH", dataset, metric)["rows"]
    pq = get_run(runs, "PQ", dataset, metric)["rows"]
    hnsw = get_run(runs, "HNSW", dataset, metric)["rows"]

    panels = [
        ("LSH — nbits", "LSH", "nbits", lsh),
        ("PQ — nbits (m=32)", "PQ", "nbits", [r for r in pq if r["params"]["m"] == 32]),
        ("PQ — m (nbits=10)", "PQ", "m", [r for r in pq if r["params"]["nbits"] == 10]),
        ("HNSW — efSearch (M=32)", "HNSW", "efSearch", [r for r in hnsw if r["params"]["M"] == 32]),
        ("HNSW — M (efSearch=64)", "HNSW", "M", [r for r in hnsw if r["params"]["efSearch"] == 64]),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(15, 3.2), sharey=True)

    for ax, (title, method, key, rows) in zip(axes, panels):
        rows = sorted(rows, key=lambda r: r["params"][key])
        xs = [r["params"][key] for r in rows]

        ax.plot(xs, [r["recall"] for r in rows], color=COLOURS[method], linewidth=2,
                marker="o", markersize=6, markeredgecolor="white", markeredgewidth=0.8)

        logx = max(xs) / min(xs) > 8
        style_axes(ax, key, "recall@10" if ax is axes[0] else "", title, logx=logx)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(x) for x in xs])
        ax.minorticks_off()

    fig.suptitle(f"Parameter sensitivity — {DATASET_LABEL[dataset]} data, {metric.upper() if metric == 'l2' else 'cosine'}, d=384",
                 fontsize=11, color=INK, y=1.03)
    save(fig, "fig1_parameter_sweeps.png")



def plot_tradeoffs(runs, dataset="clustered", metric="cosine"):
    """
    Report section 3: recall against memory, speed and build cost, all methods on one dataset
    """
    axes_spec = [
        ("size", "index size (MB, log)", False, "flat_size"),
        ("speedup", "speedup vs Flat (×, log)", True, None),
        ("build", "build time (s, log)", False, "flat_build"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    flat = get_run(runs, "LSH", dataset, metric)["flat"]

    for ax, (key, xlabel, higher_better, ref) in zip(axes, axes_spec):
        if key == "speedup":
            ax.axvline(1, color=FLAT, linewidth=1.2, linestyle=":")
            ax.text(1, 0.02, " Flat", fontsize=7, color=INK_MUTED, va="bottom")
        elif key == "size":
            ax.axvline(flat["size"], color=FLAT, linewidth=1.2, linestyle=":")
            ax.text(flat["size"], 0.02, " Flat", fontsize=7, color=INK_MUTED, va="bottom", ha="right")

        for method in METHODS:
            rows = get_run(runs, method, dataset, metric)["rows"]

            ax.scatter([r[key] for r in rows], [r["recall"] for r in rows],
                       s=16, color=COLOURS[method], alpha=0.35, linewidths=0, zorder=2)

            front = frontier(rows, key, higher_better)
            ax.plot([r[key] for r in front], [r["recall"] for r in front], color=COLOURS[method],
                    linewidth=2, marker="o", markersize=6, markeredgecolor="white",
                    markeredgewidth=0.8, label=method, zorder=3)

            best = max(rows, key=lambda r: r["recall"])
            ax.annotate(method, (best[key], best["recall"]), textcoords="offset points",
                        xytext=(5, 4), fontsize=8, color=INK)

        style_axes(ax, xlabel, "recall@10" if ax is axes[0] else "", "", logx=True)

    axes[0].legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK_MUTED)
    fig.suptitle(f"Method trade-offs — {DATASET_LABEL[dataset]} data, {metric.upper() if metric == 'l2' else 'cosine'} "
                 f"(faint points: all configs; line: best achievable)", fontsize=11, color=INK, y=1.03)
    save(fig, "fig2_method_tradeoffs.png")



def plot_dataset_effect(runs, metric="cosine"):
    """
    Report section 4: recall against speedup on every dataset, cosine throughout
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)

    for ax, dataset in zip(axes, DATASETS):
        ax.axvline(1, color=FLAT, linewidth=1.2, linestyle=":")
        ax.text(1, 0.02, " Flat", fontsize=7, color=INK_MUTED, va="bottom")

        for method in METHODS:
            rows = get_run(runs, method, dataset, metric)["rows"]

            ax.scatter([r["speedup"] for r in rows], [r["recall"] for r in rows],
                       s=16, color=COLOURS[method], alpha=0.35, linewidths=0, zorder=2)

            front = frontier(rows, "speedup", True)
            ax.plot([r["speedup"] for r in front], [r["recall"] for r in front], color=COLOURS[method],
                    linewidth=2, marker="o", markersize=6, markeredgecolor="white",
                    markeredgewidth=0.8, label=method, zorder=3)

            best = max(rows, key=lambda r: r["recall"])
            ax.annotate(method, (best["speedup"], best["recall"]), textcoords="offset points",
                        xytext=(5, 4), fontsize=8, color=INK)

        style_axes(ax, "speedup vs Flat (×, log)", "recall@10" if ax is axes[0] else "",
                   DATASET_LABEL[dataset], logx=True)

    axes[0].legend(frameon=False, fontsize=8, loc="lower left", labelcolor=INK_MUTED)
    fig.suptitle("Effect of data — recall against speedup, cosine, d=384", fontsize=11, color=INK, y=1.03)
    save(fig, "fig3_dataset_effect.png")



def main():
    runs = parse_results()
    print(f"parsed {len(runs)} runs, {sum(len(r['rows']) for r in runs)} configurations")

    plot_parameter_sweeps(runs)
    plot_tradeoffs(runs)
    plot_dataset_effect(runs)



if __name__ == "__main__":
    main()
