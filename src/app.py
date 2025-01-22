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

'''
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
'''

# We'll define some convenience lists for year/month selections
YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
MONTHS = list(range(1,13))

# Day-of-week options
DAY_OPTIONS = [
    {"label": "Monday",       "value": "MON"},
    {"label": "Tuesday",      "value": "TUE"},
    {"label": "Wednesday",    "value": "WED"},
    {"label": "Thursday",     "value": "THU"},
    {"label": "Friday",       "value": "FRI"},
    {"label": "Saturday",     "value": "SAT"},
    {"label": "Sunday",       "value": "SUN"},
    {"label": "All Weekdays", "value": "WEEKDAY"},
    {"label": "All Weekends", "value": "WEEKEND"},
]

# Helper function: produce a set of (year, month) dropdowns for start & end
def period_controls(prefix):
    return html.Div([
        html.Label(f"{prefix} Start Year:"),
        dcc.Dropdown(
            id=f"{prefix.lower()}-start-year",
            options=[{"label": y, "value": y} for y in YEARS],
            value=2018,  # default
            clearable=False,
            style={"width":"100px"}
        ),
        html.Label("Start Month:"),
        dcc.Dropdown(
            id=f"{prefix.lower()}-start-month",
            options=[{"label": m, "value": m} for m in MONTHS],
            value=1,
            clearable=False,
            style={"width":"80px"}
        ),
        html.Label(f"{prefix} End Year:"),
        dcc.Dropdown(
            id=f"{prefix.lower()}-end-year",
            options=[{"label": y, "value": y} for y in YEARS],
            value=2019,
            clearable=False,
            style={"width":"100px"}
        ),
        html.Label("End Month:"),
        dcc.Dropdown(
            id=f"{prefix.lower()}-end-month",
            options=[{"label": m, "value": m} for m in MONTHS],
            value=12,
            clearable=False,
            style={"width":"80px"}
        ),
    ], style={"display":"inline-block", "marginRight":"20px"})


app.layout = html.Div([
    html.H1("ATR Traffic Dashboard"),

    # Map
    dcc.Graph(id="map-graph", figure=fig_map),

    # Controls for period 1 and period 2
    html.Div([
        period_controls("Period1"),
        period_controls("Period2"),
    ]),

    # Day-of-week selection
    html.Div([
        html.Label("Day of Week:"),
        dcc.Dropdown(
            id="day-of-week",
            options=DAY_OPTIONS,
            value="WEEKDAY",  # default
            clearable=False,
            style={"width":"150px"}
        ),
    ], style={"marginTop":"20px", "marginBottom":"20px"}),

    html.H3("Selected Location Plots (Interactive Filter)"),

    dcc.Graph(id="traffic-graph", figure=fig_traffic),
])

@app.callback(
    Output("traffic-graph", "figure"),
    [
        Input("map-graph", "clickData"),
        Input("period1-start-year", "value"),
        Input("period1-start-month", "value"),
        Input("period1-end-year", "value"),
        Input("period1-end-month", "value"),
        Input("period2-start-year", "value"),
        Input("period2-start-month", "value"),
        Input("period2-end-year", "value"),
        Input("period2-end-month", "value"),
        Input("day-of-week", "value"),
    ]
)
def update_traffic_plot(
    click_data,
    p1_start_yr, p1_start_mo, p1_end_yr, p1_end_mo,
    p2_start_yr, p2_start_mo, p2_end_yr, p2_end_mo,
    day_choice
):
    """
    1) Retrieve location from map click.
    2) Determine two periods from user input.
    3) Filter df accordingly.
    4) Plot two subplots (one for direction1, one for direction2).
    """

    # If no map click
    if click_data is None:
        return go.Figure().update_layout(title="Select a location on the map.")

    location_id = click_data["points"][0]["customdata"]

    # 1) Build sets or lists of year-month for each period
    #    We'll define a function that returns a boolean mask if the row is in that period
    #    For simplicity, we combine year+month into a single integer like (year * 100 + month)
    df["YearMonth"] = df["Year"]*100 + df["DateTime"].dt.month

    period1_start = p1_start_yr * 100 + p1_start_mo
    period1_end   = p1_end_yr   * 100 + p1_end_mo
    period2_start = p2_start_yr * 100 + p2_start_mo
    period2_end   = p2_end_yr   * 100 + p2_end_mo

    # Create a new "Period" column for each row: "Period1", "Period2", or None if not in either.
    # We'll do it in a copy to avoid messing the global df.
    df_local = df.copy()
    df_local["Period"] = None

    mask_p1 = (df_local["YearMonth"] >= period1_start) & (df_local["YearMonth"] <= period1_end)
    mask_p2 = (df_local["YearMonth"] >= period2_start) & (df_local["YearMonth"] <= period2_end)

    df_local.loc[mask_p1, "Period"] = "Period1"
    df_local.loc[mask_p2, "Period"] = "Period2"

    # 2) Filter to Lane=None, direction in [NB,SB,EB,WB], location ID
    dirs = ["NB","SB","EB","WB"]
    sub = df_local[
        (df_local["Lane"].isna()) &
        (df_local["Direction"].isin(dirs)) &
        (df_local["LocationID"] == location_id) &
        (df_local["Period"].notna())
    ].copy()

    if sub.empty:
        return go.Figure().update_layout(title=f"No data for {location_id} in selected periods")

    # 3) Handle day-of-week filtering
    #    We can create a column DayOfWeek = df["DateTime"].dt.dayofweek (0=Mon,...6=Sun)
    #    Then interpret user choice:
    #       MON -> 0
    #       TUE -> 1
    #       ...
    #       WEEKDAY -> 0..4
    #       WEEKEND -> 5..6
    sub["DayOfWeek"] = sub["DateTime"].dt.dayofweek  # Monday=0, Sunday=6

    def day_filter(df_, choice):
        if choice == "WEEKDAY":
            return df_[df_["DayOfWeek"] <= 4]
        elif choice == "WEEKEND":
            return df_[(df_["DayOfWeek"] == 5) | (df_["DayOfWeek"] == 6)]
        else:
            # Specific day? We'll map MON->0, TUE->1, ...
            mapping = {
                "MON": 0, "TUE":1, "WED":2, "THU":3,
                "FRI":4, "SAT":5, "SUN":6
            }
            wanted = mapping.get(choice, None)
            if wanted is None:
                return df_  # fallback if choice is unrecognized
            return df_[df_["DayOfWeek"] == wanted]

    sub = day_filter(sub, day_choice)

    if sub.empty:
        return go.Figure().update_layout(title=f"No data after day-of-week filter for {location_id}")

    # 4) Now we have sub with columns: Period in {Period1, Period2}, plus directions
    #    We'll produce a two-subplot figure with fill bands for each Period.
    #    Similar to the earlier approach, but the "Period" column is now "Period1" / "Period2" 
    #    instead of "2018–2019" / "2023–2024".
    fig = plot_two_subplots_mean_iqr(sub)
    return fig


#####################
# 4) The Two-Subplot Function
#####################

def plot_two_subplots_mean_iqr(df_sub):
    """
    Produces a 1x2 subplot figure. Each subplot = one direction.
    X-axis = Hour, Y-axis = Volume
    Data from Period1 or Period2 => fill-between Q1..Q3 plus mean line.
    """

    # Force hour from DateTime, or if you already have df_sub["Hour"]:
    df_sub["Hour"] = df_sub["DateTime"].dt.hour

    # Identify which directions
    directions = df_sub["Direction"].dropna().unique()
    directions = [d for d in directions if d in ["NB","SB","EB","WB"]]

    if len(directions) == 0:
        return go.Figure().update_layout(title="No recognized directions")

    # If more than 2 directions, take first 2
    directions = directions[:2]

    # Make subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[f"Dir: {directions[0]}", f"Dir: {directions[1]}" if len(directions)>1 else ""],
        shared_yaxes=True,
        horizontal_spacing=0.1
    )

    # We'll define two "named periods": "Period1" and "Period2" 
    # and color them differently
    color_map = {"Period1": "blue", "Period2": "red"}

    # For each direction => filter => group by (Period, Hour) => mean, Q1, Q3 => pivot => fill
    for i, direction in enumerate(directions):
        c = i+1  # col index
        dir_df = df_sub[df_sub["Direction"] == direction].copy()
        if dir_df.empty:
            continue

        # Group
        agg = (
            dir_df.groupby(["Period","Hour"])["Volume"]
                  .agg(
                      mean="mean",
                      Q1=lambda x: x.quantile(0.25),
                      Q3=lambda x: x.quantile(0.75)
                  )
                  .reset_index()
        )

        # Pivot wide => columns like mean_Period1, mean_Period2, Q1_Period1, ...
        wide = agg.pivot(index="Hour", columns="Period", values=["mean","Q1","Q3"])
        wide.columns = [f"{lvl0}_{lvl1}" for lvl0,lvl1 in wide.columns]
        wide = wide.reset_index()

        # For each period in [Period1, Period2], add fill + mean line
        for period in ["Period1","Period2"]:
            if f"mean_{period}" not in wide.columns:
                continue

            # Q1 line
            fig.add_trace(go.Scatter(
                name=f"{period} (Q1)",
                x=wide["Hour"],
                y=wide.get(f"Q1_{period}", pd.Series()),
                mode="lines",
                line=dict(width=0),
                showlegend=False
            ), row=1, col=c)

            # Q3 fill
            fig.add_trace(go.Scatter(
                name=f"{period} IQR",
                x=wide["Hour"],
                y=wide.get(f"Q3_{period}", pd.Series()),
                mode="lines",
                fill="tonexty",
                line=dict(color=color_map[period], width=0),
                opacity=0.2
            ), row=1, col=c)

            # mean line
            fig.add_trace(go.Scatter(
                name=f"{period} Mean",
                x=wide["Hour"],
                y=wide.get(f"mean_{period}", pd.Series()),
                mode="lines",
                line=dict(color=color_map[period])
            ), row=1, col=c)

    fig.update_layout(
        title="IQR + Mean by Direction, Two Periods",
        hovermode="x unified",
        template="plotly_white"
    )

    fig.update_xaxes(title_text="Hour of Day", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=1, col=1)

    if len(directions)>1:
        fig.update_xaxes(title_text="Hour of Day", row=1, col=2)

    return fig

'''
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
    fig = plot_iqr_bands(sub, atr_name, location_id)
    return fig

def plot_iqr_bands(df_sub, name, location_id):
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
        title=f"ATR ID: {location_id} Weekday Traffic: Mean & IQR (2018–2019 vs 2023–2024) for {name}",
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
'''

if __name__ == "__main__":
    app.run_server(debug=True)
