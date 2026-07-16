"""
04_dashboard.py — GridPulse Renewable Energy & EV Infrastructure Analytics
Minimalist dark dashboard inspired by Airbnb design system.
http://127.0.0.1:8051/
"""

import os
import sqlite3
import pandas as pd
import numpy as np
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

TABLES_DIR = "gridpulse_energy_analytics/outputs/tables"
DB_PATH = "gridpulse_energy_analytics/data/gridpulse.db"

def load_data():
    data = {}
    for name in ["q1_grid_frequency_deviation_ewma", "q2_renewable_curtailment_weather",
                 "q3_state_re_integration_scorecard", "q4_ev_charging_spatial_gap",
                 "q5_peak_demand_generation_gap", "dbscan_ev_charging_deserts",
                 "p_median_optimal_hub_locations", "var_granger_summary"]:
        path = os.path.join(TABLES_DIR, f"{name}.csv")
        if os.path.exists(path):
            data[name] = pd.read_csv(path)
        else:
            data[name] = pd.DataFrame()
    if os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        data["states"] = pd.read_sql_query("SELECT * FROM states", conn)
        conn.close()
    return data

data = load_data()

C = {
    "bg":     "#000000",
    "card":   "#111111",
    "border": "#222222",
    "text":   "#ffffff",
    "muted":  "#717171",
    "accent": "#FF385C",
    "green":  "#00A699",
    "chart_bg": "#111111",
}

PLOT_TEMPLATE = dict(
    plot_bgcolor=C["chart_bg"],
    paper_bgcolor=C["chart_bg"],
    font=dict(family="Cereal, Helvetica Neue, Arial, sans-serif", color=C["text"], size=13),
    xaxis=dict(gridcolor="#222222", zerolinecolor="#222222"),
    yaxis=dict(gridcolor="#222222", zerolinecolor="#222222"),
    margin=dict(l=40, r=20, t=50, b=40),
    hoverlabel=dict(bgcolor="#222222", font_color="#ffffff"),
)

FONT = "'Cereal', 'Helvetica Neue', Arial, sans-serif"
app = dash.Dash(__name__, title="GridPulse — Energy & EV Analytics")

def kpi(label, value, note=""):
    return html.Div([
        html.Span(label, style={"color": C["muted"], "fontSize": "12px", "letterSpacing": "0.5px", "textTransform": "uppercase"}),
        html.Div(value, style={"fontSize": "28px", "fontWeight": "600", "color": C["text"], "margin": "6px 0 2px"}),
        html.Span(note, style={"fontSize": "12px", "color": C["muted"]}) if note else None,
    ], style={"padding": "24px 0", "flex": "1", "textAlign": "center"})

def section_title(text):
    return html.H3(text, style={"fontSize": "16px", "fontWeight": "600", "color": C["text"], "margin": "0 0 16px", "letterSpacing": "-0.2px"})

def card(children, **kwargs):
    style = {"backgroundColor": C["card"], "borderRadius": "12px", "padding": "24px", "marginBottom": "16px", "border": f"1px solid {C['border']}"}
    style.update(kwargs.get("style", {}))
    return html.Div(children, style=style)

div = lambda x: html.Div(style={"width": "1px", "backgroundColor": C["border"], "alignSelf": "stretch"})

app.layout = html.Div(style={"backgroundColor": C["bg"], "color": C["text"], "minHeight": "100vh", "fontFamily": FONT, "padding": "32px 0", "width": "100%"}, children=[
    html.Div(style={"maxWidth": "1200px", "margin": "0 auto", "padding": "0 48px"}, children=[
        html.Div([
            html.H1("GridPulse", style={"fontSize": "24px", "fontWeight": "700", "margin": "0", "letterSpacing": "-0.5px"}),
            html.P("Renewable Energy Grid Analytics & EV Infrastructure  ·  28 States", style={"color": C["muted"], "fontSize": "14px", "margin": "4px 0 0"})
        ], style={"marginBottom": "32px"}),

        card([html.Div([
            kpi("RE Capacity Monitored", "112.4 GW", "Solar & wind across 28 states"),
            div(None),
            kpi("Curtailment Revenue Loss", "₹606 Cr", "Peak solar oversupply — RJ & TN"),
            div(None),
            kpi("Solar Peak Violation Rate", "20.5%", "Gujarat statutory band breach"),
            div(None),
            kpi("Optimal EV Hubs Selected", "25", "p-Median facility location"),
        ], style={"display": "flex", "alignItems": "center"})]),

        dcc.Tabs(id="tabs", value="t1", style={"marginBottom": "24px"}, children=[
            dcc.Tab(label="Overview", value="t1"),
            dcc.Tab(label="Grid Stability", value="t2"),
            dcc.Tab(label="State Scorecard", value="t3"),
            dcc.Tab(label="EV Charging Map", value="t4"),
            dcc.Tab(label="Optimal Hubs", value="t5"),
        ], colors={"border": C["bg"], "primary": C["accent"], "background": C["bg"]}),

        html.Div(id="content")
    ])
])

app.index_string = '''<!DOCTYPE html>
<html>
<head>{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
<style>
html, body {
    background-color: #000000 !important;
    margin: 0 !important;
    padding: 0 !important;
    color: #ffffff !important;
}
.tab{background:transparent!important;color:#717171!important;border:none!important;padding:10px 20px!important;font-size:13px!important;font-weight:500!important;font-family:'Helvetica Neue',Arial,sans-serif!important;letter-spacing:0.2px}
.tab--selected{color:#fff!important;border-bottom:2px solid #FF385C!important}
.tab:hover{color:#fff!important}
._dash-debug-menu, .dash-debug-menu, ._dash-footer, #_dash-global-error-container { display: none !important; }
</style>
</head>
<body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body>
</html>'''

@app.callback(Output("content", "children"), Input("tabs", "value"))
def render(tab):

    if tab == "t1":
        df = data.get("q5_peak_demand_generation_gap", pd.DataFrame())
        if not df.empty:
            fig = px.bar(df, x="state_name", y="frequency_violation_rate_pct", color="diurnal_period", barmode="group",
                         color_discrete_map={"SOLAR PEAK (11 AM - 3 PM)": C["accent"], "EVENING PEAK (6 PM - 10 PM)": "#484848", "BASE / OFF-PEAK HOURS": C["green"]},
                         labels={"frequency_violation_rate_pct": "Violation rate (%)", "state_name": "", "diurnal_period": ""})
            fig.update_layout(**PLOT_TEMPLATE, title=None, legend=dict(orientation="h", y=1.08, x=0, bgcolor="rgba(0,0,0,0)"))
        else:
            fig = go.Figure()

        return html.Div([
            card([section_title("Grid frequency statutory violation rate by diurnal period"), dcc.Graph(figure=fig, config={"displayModeBar": False})]),
            card([
                section_title("Key findings"),
                html.P("Granger causality tests across 4 lags confirm that sudden renewable curtailment ramps statistically cause grid frequency instability (F > 36, p < 0.001). During solar peak hours, Gujarat's violation rate surges to 20.5% — a 2 GWh BESS buffer would absorb ramp shocks and prevent statutory grid frequency collapse.", style={"color": C["muted"], "lineHeight": "1.7", "fontSize": "14px", "margin": "0 0 12px"}),
                html.P("DBSCAN clustering identified 21 distinct highway EV charging deserts. p-Median optimization selected 25 strategically optimal ultra-fast hub locations minimizing total driver deviation distance nationwide.", style={"color": C["muted"], "lineHeight": "1.7", "fontSize": "14px", "margin": "0"}),
            ])
        ])

    elif tab == "t2":
        df = data.get("q1_grid_frequency_deviation_ewma", pd.DataFrame())
        if not df.empty:
            fig = px.bar(df, x="state_name", y="total_duration_hours", color="stability_status",
                         color_discrete_map={"STATUTORY VIOLATION (>0.05 Hz)": C["accent"], "WARNING BAND (>0.03 Hz)": "#484848"},
                         labels={"total_duration_hours": "Hours of instability", "state_name": "", "stability_status": ""})
            fig.update_layout(**PLOT_TEMPLATE, title=None, legend=dict(orientation="h", y=1.08, x=0, bgcolor="rgba(0,0,0,0)"))
        else:
            fig = go.Figure()
        return card([section_title("Cumulative hours of grid frequency instability by state"), dcc.Graph(figure=fig, config={"displayModeBar": False})])

    elif tab == "t3":
        df_sc = data.get("q3_state_re_integration_scorecard", pd.DataFrame())
        if not df_sc.empty:
            fig = px.scatter(df_sc, x="re_penetration_ratio_pct", y="state_curtail_pct", size="total_re_capacity_mw",
                             color="region", hover_name="state_name",
                             color_discrete_sequence=[C["accent"], C["green"], "#484848", "#717171"],
                             labels={"re_penetration_ratio_pct": "RE penetration (%)", "state_curtail_pct": "Curtailment rate (%)", "region": ""})
            fig.update_layout(**PLOT_TEMPLATE, title=None, legend=dict(orientation="h", y=1.08, x=0, bgcolor="rgba(0,0,0,0)"))
        else:
            fig = go.Figure()

        df_c = data.get("q2_renewable_curtailment_weather", pd.DataFrame())
        if not df_c.empty:
            fig2 = px.bar(df_c, x="state_name", y="total_curtailed_mwh", color="curtailment_cause", facet_col="source_type",
                          color_discrete_sequence=[C["accent"], C["green"], "#484848", "#717171"],
                          labels={"total_curtailed_mwh": "Curtailed MWh", "state_name": "", "curtailment_cause": ""})
            fig2.update_layout(**PLOT_TEMPLATE, title=None, height=400, legend=dict(orientation="h", y=1.12, x=0, bgcolor="rgba(0,0,0,0)"))
        else:
            fig2 = go.Figure()

        return html.Div([
            card([section_title("RE penetration vs curtailment rate by state"), dcc.Graph(figure=fig, config={"displayModeBar": False})]),
            card([section_title("Curtailment root-cause decomposition"), dcc.Graph(figure=fig2, config={"displayModeBar": False})])
        ])

    elif tab == "t4":
        df = data.get("q4_ev_charging_spatial_gap", pd.DataFrame())
        if not df.empty:
            fig = px.scatter_map(df, lat="lat", lon="lon", color="grid_stress_index", size="unmet_daily_trips_gap",
                                    hover_name="highway_name", hover_data=["segment_km", "critical_demand"],
                                    color_continuous_scale=["#222222", C["accent"]], zoom=4, height=520)
            fig.update_layout(map_style="carto-darkmatter", **{k: PLOT_TEMPLATE[k] for k in ["paper_bgcolor", "font", "margin"]})
        else:
            fig = go.Figure()
        return card([section_title("Highway EV charging demand-supply gap"), dcc.Graph(figure=fig, config={"displayModeBar": False})])

    elif tab == "t5":
        df = data.get("p_median_optimal_hub_locations", pd.DataFrame())
        if not df.empty:
            tbl = dash_table.DataTable(
                columns=[{"name": c, "id": c} for c in df.columns],
                data=df.to_dict("records"),
                page_size=15, sort_action="native", filter_action="native",
                style_header={"backgroundColor": "#111111", "color": "#ffffff", "fontWeight": "600", "borderBottom": f"1px solid {C['border']}", "fontSize": "12px", "textTransform": "uppercase", "letterSpacing": "0.4px"},
                style_cell={"backgroundColor": "#111111", "color": "#e0e0e0", "border": "none", "padding": "10px 14px", "fontSize": "13px", "fontFamily": FONT},
                style_data_conditional=[{"if": {"column_id": "hub_rank", "filter_query": "{hub_rank} le 5"}, "color": C["green"], "fontWeight": "600"}]
            )
        else:
            tbl = html.P("No data", style={"color": C["muted"]})
        return card([section_title("Top 25 optimal ultra-fast charging hub locations"), tbl])

if __name__ == "__main__":
    app.run(debug=True, dev_tools_ui=False, dev_tools_props_check=False, port=8051)
