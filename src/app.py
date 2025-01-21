import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import logging
import sys
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils import load_metadata_pandas, load_processed_pandas

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)

df_traffic = load_processed_pandas()
df_meta = load_metadata_pandas()

# New dataframe which is inner join of metadata and long processed
# and filter df_meta to only have valid ATR locations we can generate
# plots for.
df = pd.merge(df_traffic, df_meta, on='LocationID', how='inner')
valid_ids = df['LocationID'].unique()
df_meta = df_meta[df_meta['LocationID'].isin(valid_ids)].copy()

# 3) Create a Plotly map figure
#    We'll store the location ID in customdata so we can retrieve it on clicks
fig_map = go.Figure(
    data=go.Scattermap(
        lat=df_meta['LAT'],
        lon=df_meta['LONGTD'],
        text=df_meta['ATR_NAME'],
        customdata=df_meta['LocationID'],  # store the ID for callback usage
        mode='markers',
        marker=dict(
            color='black', 
            size=25, 
        ),
        selected=dict(
            marker=dict(color='red', size=30)
        ),
    ), 
    layout=dict(
        map=dict(
            style="streets",
            center=dict(lat=45.5152, lon=-122.6784),
            zoom=9,
        ),
        height=600,
        width=600,
        clickmode='event+select',  # Enable selection on click
        margin=dict(l=0, r=0, t=0, b=0)
    )
)


# 4) Create a placeholder figure for the traffic plot
fig_traffic = go.Figure()
fig_traffic.update_layout(
    title="Traffic Volume for Selected Location"
)

# 5) Build the Dash app layout
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("ATR Traffic Dashboard"),
    html.Div(
        style={"display": "flex", "flex-grow": 1}, 
        children=[
            dcc.Graph(id="map-graph", figure=fig_map),
            html.Div(
                style={"display": "flex", "flex-grow": 1}, 
                children=[
                    #html.H3("Selected Location Plots"),
                    dcc.Graph(id="traffic-graph", figure=fig_traffic)
                ])
        ])
])

# 6) Define callback: when user clicks a marker on the map,
#    we retrieve the location ID from `clickData` and filter df_traffic
@app.callback(
    Output("traffic-graph", "figure"),
    Input("map-graph", "clickData")
)
def update_traffic_plot(click_data):
    if click_data is None:
        return go.Figure().update_layout(title="Select a location on the map")

    location_id = click_data["points"][0]["customdata"]
    atr_name = click_data["points"][0]["text"]
    # Filter the traffic DataFrame for that location, ignoring lane-level data & focusing on cardinal directions
    keep_years = [2018, 2019, 2023, 2024]
    dirs = ['NB','SB','EB','WB']
    sub = df[
        (df['Lane'].isna()) &
        (df['Direction'].notna()) &
        (df['Direction'].isin(dirs)) &
        (df['LocationID'] == location_id) &
        (df['Year'].isin(keep_years))
    ].copy()

    if sub.empty:
        return go.Figure().update_layout(title=f"No data for location ID = {location_id}")
    fig = plot_iqr_bands(sub, atr_name)
    return fig

def plot_iqr_bands(df_sub, name):
    """
    df_sub: A DataFrame with [LocationID, Direction, Year, Hour, Weekend, Volume].
    Returns a go.Figure with 2 subplots: (Direction1, Direction2),
    each showing Mean & IQR for 2018–2019 vs 2023–2024 (only Weekday data).
    If only one direction is found, it just populates one subplot.
    """

    # 1) Weekday-only
    df_weekday = df_sub[df_sub["Weekend"] == False].copy()

    # 2) Determine which directions exist in this subset
    #    e.g. NB, SB, EB, WB. We'll pick the first two if there's more than two
    directions = df_weekday["Direction"].dropna().unique()
    directions = [d for d in directions if d in ["NB","SB","EB","WB"]]

    if len(directions) == 0:
        # No recognized directions found
        fig_empty = go.Figure()
        fig_empty.update_layout(title="No valid directional data")
        return fig_empty

    # If more than 2 directions, just take the first 2 for now
    directions = directions[:2]

    # 3) Make subplots: 1 col, 2 rows
    #    If only 1 direction, we'll just leave the second subplot mostly blank
    fig = make_subplots(
        rows=2, cols=1, 
        subplot_titles=[f"Direction: {directions[0]}", 
                        f"Direction: {directions[1]}" if len(directions) > 1 else ""],
        shared_xaxes=False,
        shared_yaxes=False,
        vertical_spacing=0.25
    )

    # 4) For each direction, compute the stats and add traces
    for i, direction in enumerate(directions):
        col_index = 1
        row_index = i + 1  # subplot col: 1 or 2
        # Filter to that direction
        sub_dir = df_weekday[df_weekday["Direction"] == direction].copy()

        if sub_dir.empty:
            continue

        # Define period
        sub_dir["Period"] = np.where(
            sub_dir["Year"].isin([2018,2019]),
            "2018–2019",
            "2023–2024"
        )

        # Group (Period, Hour) -> mean, Q1, Q3
        agg = (
            sub_dir.groupby(["Period","Hour"])["Volume"]
                  .agg(
                      mean="mean",
                      Q1=lambda x: x.quantile(0.25),
                      Q3=lambda x: x.quantile(0.75)
                  )
                  .reset_index()
        )

        # Pivot wide
        wide = agg.pivot(index="Hour", columns="Period", values=["mean","Q1","Q3"])
        # Flatten columns
        wide.columns = [f"{lvl0}_{lvl1}" for lvl0,lvl1 in wide.columns]
        # e.g. ["mean_2018–2019", "mean_2023–2024", "Q1_2018–2019", ...

        wide = wide.reset_index()  # so "Hour" is a column

        # Add fill bands + mean lines for each period
        # --- 2018–2019 ---
        fig.add_trace(go.Scatter(
            name="2018–2019 (Q1)",
            x=wide["Hour"], 
            y=wide.get("Q1_2018–2019", pd.Series()),  # fallback if col missing
            mode="lines",
            line=dict(width=0),
            showlegend=False
        ), row=row_index, col=col_index)

        fig.add_trace(go.Scatter(
            name="2018–2019 IQR",
            x=wide["Hour"],
            y=wide.get("Q3_2018–2019", pd.Series()),
            mode="lines",
            fill="tonexty",
            showlegend=i==0,
            line=dict(color="blue", width=0),
            opacity=0.1
        ), row=row_index, col=col_index)

        fig.add_trace(go.Scatter(
            name="2018–2019 Mean",
            x=wide["Hour"],
            y=wide.get("mean_2018–2019", pd.Series()),
            mode="lines",
            showlegend=i==0,
            line=dict(color="blue")
        ), row=row_index, col=col_index)

        # --- 2023–2024 ---
        fig.add_trace(go.Scatter(
            name="2023–2024 (Q1)",
            x=wide["Hour"],
            y=wide.get("Q1_2023–2024", pd.Series()),
            mode="lines",
            line=dict(width=0),
            showlegend=False
        ), row=row_index, col=col_index)

        fig.add_trace(go.Scatter(
            name="2023–2024 IQR",
            x=wide["Hour"],
            y=wide.get("Q3_2023–2024", pd.Series()),
            mode="lines",
            fill="tonexty",
            showlegend=i==0,
            line=dict(color="red", width=0),
            opacity=0.1
        ), row=row_index, col=col_index)

        fig.add_trace(go.Scatter(
            name="2023–2024 Mean",
            x=wide["Hour"],
            y=wide.get("mean_2023–2024", pd.Series()),
            mode="lines",
            showlegend=i==0,
            line=dict(color="red")
        ), row=row_index, col=col_index)

    # 5) Layout
    fig.update_layout(
        title=f"Weekday Traffic: Mean & IQR (2018–2019 vs 2023–2024) for {name}",
        hovermode="x unified",
        template="plotly_white",
        height=600,
        width=1000,
    )

    # Shared axis labels
    fig.update_xaxes(title_text="Hour of Day", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=1, col=1)

    # If two directions, add x-label for second subplot
    if len(directions) > 1:
        fig.update_xaxes(title_text="Hour of Day", row=1, col=2)

    return fig

if __name__ == "__main__":
    app.run_server(debug=True)
