"""Export gait results to CSV / XLSX and figures to PNG."""
import csv

import openpyxl


def export_results_csv(results, path):
    """Write spatiotemporal parameters as a (parameter, value) CSV."""
    st = results.get("spatiotemporal", {})
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["parameter", "value"])
        for k, v in st.items():
            writer.writerow([k, v])


def export_results_xlsx(results, path):
    """Write spatiotemporal + per-joint mean angle curves to a multi-sheet XLSX."""
    wb = openpyxl.Workbook()
    st_sheet = wb.active
    st_sheet.title = "spatiotemporal"
    st_sheet.append(["parameter", "value"])
    for k, v in results.get("spatiotemporal", {}).items():
        st_sheet.append([k, v])

    ang = wb.create_sheet("joint_angles_mean")
    means = results.get("joint_angles_mean", {})
    keys = list(means.keys())
    ang.append(["percent_cycle"] + keys)
    n = max((len(v) for v in means.values()), default=0)
    for i in range(n):
        ang.append([i] + [means[k][i] for k in keys])
    wb.save(path)


def export_figure_png(fig, path):
    """Save a matplotlib Figure to PNG."""
    fig.savefig(path, dpi=120)
