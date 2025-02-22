import polars as pl
from traffic_data_analysis.plots import plot_hourly_quartiles_by_years, missing_data_visual
from traffic_data_analysis.utils import load_directional_hourly

def all_yearly_plots(df_traffic: pl.DataFrame, df_meta: pl.DataFrame):
    grouped = df_traffic.group_by(['LocationID', 'Direction', 'Weekend'])
    files = []
    for info, group in grouped:
        filename = plot_hourly_quartiles_by_years(info, group, df_meta)
        files.append(filename)
    return files

def build_subfigure(files):
    subfig = lambda filename : "\\begin{{subfigure}}[b]{{0.45\\textwidth}}\n\t\t\\includegraphics[width=\\textwidth]{{{}}}\n\t\\end{{subfigure}}".format(filename)

    return f"\\begin{{figure}}[htbp]\n\t\\centering\n\t{subfig(files[0])}\n\t\\hfill\n\t{subfig(files[1])}\n\n\t{subfig(files[2])}\n\t\\hfill\n\t{subfig(files[3])}\n\\end{{figure}}"

df_traffic, df_meta = load_directional_hourly()
files = all_yearly_plots(df_traffic, df_meta)
files.sort()
print("\n\n".join([build_subfigure(files[i:i+4]) for i in range(0, len(files), 4)]))
    
missing_data_visual(df_traffic, df_meta)
