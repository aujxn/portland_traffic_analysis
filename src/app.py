import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State, MATCH, ALL
import pandas as pd
import logging
import sys
import numpy as np
import plotly.graph_objects as go
import plotly.colors as pc

from utils import load_metadata_pandas, load_processed_pandas

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)

MAX_TABS = 5

df_traffic = load_processed_pandas()
df_meta = load_metadata_pandas()

# New dataframe which is inner join of metadata and long processed
# and filter df_meta to only have valid ATR locations we can generate
# plots for.
df = pd.merge(df_traffic, df_meta, on='LocationID', how='inner')
valid_ids = df['LocationID'].unique()
df_meta = df_meta[df_meta['LocationID'].isin(valid_ids)].copy()

def trace_form(idx):
# Some form fields
    return [
        html.H1("No ATR location selected", id={"type":"selected-name", "index":idx}),
        html.H3("Select a location on the map to plot", id={"type":"selected-desc", "index":idx}),

        dcc.Checklist(
            id={"type":"trace-enabled", "index":idx},
            options=[ {"label":"Enabled","value":True} ],
            value=[],
            inline=True
        ),
        html.Br(),

        html.Label("Trace Name:"),
        html.Br(),
        dcc.Input(
            id={"type":"trace-name", "index":idx}, 
            type="text", 
            value=f"trace{idx}"
        ),
        html.Br(),
        html.Br(),

        html.Label("Date Range:", id={"type":"date-label", "index":idx}),
        html.Br(),
        dcc.DatePickerRange(
            id={"type":"trace-dates", "index":idx},
            start_date="2023-01-01",
            end_date="2023-12-31"
        ),
        html.Br(),
        html.Br(),

        html.Label("Days of Week (multi-select):"),
        html.Br(),
        dcc.Checklist(
            id={"type":"trace-days-of-week", "index":idx},
            options=[
                {"label":"Mon","value":0}, {"label":"Tue","value":1},
                {"label":"Wed","value":2}, {"label":"Thu","value":3},
                {"label":"Fri","value":4}, {"label":"Sat","value":5},
                {"label":"Sun","value":6},
            ],
            value=[0,1,2,3,4],  # default: weekdays
            inline=True
        ),
        html.Br(),

        # TODO disable invalid directions
        html.Label("Direction:"),
        html.Br(),
        dcc.RadioItems(
            id={"type":"trace-direction","index":idx},
            options=[{"label":d,"value":d,"disabled":True} for d in ["NB","SB","EB","WB"]],
            value="NB"
        ),
        html.Br(),

        html.Label("Quartiles (Lower / Upper):"),
        html.Br(),
        dcc.RangeSlider(
            0.0, 
            1.0, 
            0.05, 
            value=[0.25, 0.75], 
            id={'type':'trace-quartiles', 'index':idx}
        ),
        html.Br(),
    ]


def make_tab_content(idx):
    """
    Returns the layout (a Div) for a single tab with index `idx`.
    Includes a map + some form fields.
    We'll use pattern-matching IDs with { "type":..., "index": idx }
    so we can handle them in callbacks.
    """
    return html.Div([
        # A map
        html.Div(
            dcc.Graph(
                id={"type": "trace-map", "index": idx},
                figure=go.Figure(),  # we fill it later
                style={"height":"700px"},
                config={"scrollZoom": False}
            ),
            style={'display': 'inline-block', 'height':'600px', 'width': '40%', 'vertical-align': 'top'}
        ),
        html.Div(
            trace_form(idx),
            style={'display': 'inline-block', 'width': '60%'}
        )
    ])

app = dash.Dash(__name__)

# We create 5 tabs up front
tabs_children = []
for i in range(MAX_TABS):
    tabs_children.append(
        dcc.Tab(
            id={"type":"trace-tab","index":i},
            label=f"Trace {i}",
            value=str(i),
            children=make_tab_content(i),
        )
    )

app.layout = html.Div(
    [dcc.Store(id={"type":"trace-store","index":idx}, data=None) for idx in range(MAX_TABS)] + 
        [
            dcc.Tabs(id="trace-tabs", value="0", children=tabs_children),
            html.Button("Plot", id="btn-plot", n_clicks=0),
            dcc.Graph(id="main-fig")
        ]
)

@app.callback(
    Output({"type":"trace-store", "index":MATCH}, "data"),
    Output({"type":"trace-map","index":MATCH}, "figure"),
    Output({"type":"selected-name","index":MATCH}, "children"),
    Output({"type":"selected-desc","index":MATCH}, "children"),
    Output({"type":"trace-direction","index":MATCH}, "options"),
    #Output({"type":"trace-dates","index":MATCH}, "min_date_allowed"),
    #Output({"type":"trace-dates","index":MATCH}, "max_date_allowed"),
    Output({"type":"date-label","index":MATCH}, "children"),
    Input({"type":"trace-map","index":MATCH}, "clickData"),
    State({"type":"trace-store", "index":MATCH}, "data")
)
def map_click_update(clickData, store_data):
    name = "No ATR location selected" 
    desc = "Select a location on the map to plot"
    min_date = "2000-01-01"
    max_date = "2025-01-01"
    date_label = "Date range: "
    direction_options = [{"label":d,"value":d,"disabled":True} for d in ["NB","SB","EB","WB"]]
    ctx = dash.callback_context
    if not ctx.triggered:
        # no update yet
        # build a default figure from store_data for this tab
        #return store_data, build_map_figure(), name, desc, direction_options, min_date, max_date, date_label
        return store_data, build_map_figure(), name, desc, direction_options, date_label

    # user clicked a marker. store location in store_data for this tab
    if clickData:
        loc = clickData["points"][0]["customdata"]  # your marker's location ID
        if store_data is None:
            store_data = {}
        store_data["location_id"] = loc

        info = df_meta[df_meta["LocationID"] == loc]
        data = df[df["LocationID"]==loc]

        name = info["ATR_NAME"].values[0]
        name = f"{loc}: {name}"
        desc = info["LOCATION"].values[0]
        valid_dirs = data["Direction"].unique()
        direction_options = [{"label":d,"value":d,"disabled":d not in valid_dirs} for d in ["NB","SB","EB","WB"]]
        min_date = data["DateTime"].min()
        max_date = data["DateTime"].max()
        date_label = f"Date range (Data available from {min_date.date()} to {max_date.date()}):"


    fig = build_map_figure(location_id=store_data["location_id"] if store_data else None)
    #return store_data, fig, name, desc, direction_options, min_date, max_date, date_label
    return store_data, fig, name, desc, direction_options, date_label

def build_map_figure(location_id=None):
    # create a scattermapbox of all ATR points
    # highlight marker if location_id == ...
    # e.g. color them black, color the selected one red
    fig = go.Figure(
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
            #selected=dict(
            #    marker=dict(color='red', size=30)
            #),
        ), 
        layout=dict(
            map=dict(
                style="streets",
                center=dict(lat=45.5152, lon=-122.6784),
                zoom=9,
            ),
            showlegend=False,
            height=700,
            width=700,
            #clickmode='event+select',  # Enable selection on click
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(fixedrange=True),
            yaxis=dict(fixedrange=True)
        )
    )
    if location_id:
        # find the lat/lon of the selected
        row = df_meta[df_meta["LocationID"]==location_id]
        if not row.empty:
            sel_lat = row["LAT"].values[0]
            sel_lon = row["LONGTD"].values[0]
            fig.add_trace(go.Scattermap(
                lat=[sel_lat], lon=[sel_lon],
                marker=dict(color="red", size=30),
                mode="markers"
            ))
    return fig

@app.callback(
    Output("main-fig", "figure"),
    Input("btn-plot", "n_clicks"),
    State({"type":"trace-name", "index":ALL}, "value"), 
    State({"type":"trace-dates", "index":ALL}, "start_date"),
    State({"type":"trace-dates", "index":ALL}, "end_date"),
    State({"type":"trace-days-of-week", "index":ALL}, "value"),
    State({"type":"trace-direction","index":ALL}, "value"),
    State({'type':'trace-quartiles', 'index':ALL}, "value"),
    State({"type":"trace-enabled","index":ALL}, "value"),
    State({"type":"trace-store", "index": ALL}, "data")
)
def build_figure(n_plot, names, start_dates, end_dates, days_of_weeks, directions, quartiles, enabled, location_store):
    if n_plot < 1:
        return go.Figure().update_layout(title="No traces to plot.")

    fig = go.Figure()

    colors = pc.qualitative.Dark24
    # We'll loop over each trace config and add a line + fill band
    for i in range(MAX_TABS):
        data = [names[i], start_dates[i], end_dates[i], days_of_weeks[i], quartiles[i], location_store[i]]
        if not enabled[i] or None in data:
            print(f"Skipping plot {i} because not enabled or None is in data")
            print(data)
            continue

        # Filter df by date range, direction, days-of-week
        mask = (
            (df["LocationID"] == location_store[i]['location_id']) &
            (df["Lane"].isna()) &
            (df["DateTime"] >= start_dates[i]) &
            (df["DateTime"] <= end_dates[i]) &
            (df["Direction"] == directions[i]) &
            (df["DateTime"].dt.dayofweek.isin(days_of_weeks[i]))
        )
        sub = df[mask].copy()
        if sub.empty:
            continue

        rgb = pc.hex_to_rgb(colors[i])
        # For illustration, let's group by hour-of-day
        sub["Hour"] = sub["DateTime"].dt.hour
        grouped = sub.groupby("Hour")["Volume"]

        mean_vals = grouped.mean()
        q_low_vals = grouped.quantile(quartiles[i][0])
        q_high_vals = grouped.quantile(quartiles[i][1])

        x_hours = mean_vals.index

        # Add Q lower
        fig.add_trace(go.Scatter(
            x=x_hours, y=q_low_vals,
            line=dict(width=0),
            showlegend=False,
            name=f"{names[i]} Qlow"
        ))
        # Add Q high
        fig.add_trace(go.Scatter(
            x=x_hours, y=q_high_vals,
            fill='tonexty',  # fill area between previous trace
            line=dict(width=0),
            fillcolor= f'rgba({rgb[0]},{rgb[1]},{rgb[2]},0.2)',
            name=f"{names[i]} band"
        ))
        # Add mean line
        fig.add_trace(go.Scatter(
            x=x_hours, y=mean_vals,
            mode='lines',
            line=dict(width=2, color=colors[i]),
            name=f"{names[i]} mean"
        ))

    fig.update_layout(
        title="Custom Multi-Trace IQR Plot",
        xaxis=dict(title="Hour of Day"),
        yaxis=dict(title="Volume"),
        hovermode="x unified"
    )
    return fig

if __name__ == "__main__":
    app.run_server(debug=True)
