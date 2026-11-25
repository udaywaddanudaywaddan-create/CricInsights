"""
app.py
Presentation Layer for 'CricInsights: T20 & League Match Analytics Engine'
Made by Uday Singh | Python Project
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from db_manager import initialize_database, run_query
from analytics import (
    get_kpi_metrics,
    get_team_performance,
    get_toss_impact,
    get_top_run_scorers,
    get_top_wicket_takers,
    get_head_to_head,
    get_top_venues,
)

# -----------------------------------------------------------------------------
# Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CricInsights | Made by Uday Singh - Python Project",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #00D4FF;
        margin-bottom: 0.35rem;
    }
    .author-badge {
        display: inline-block;
        padding: 5px 15px;
        border-radius: 999px;
        font-size: 0.88rem;
        font-weight: 700;
        background: rgba(0, 212, 255, 0.14);
        color: #00D4FF;
        border: 1px solid rgba(0, 212, 255, 0.45);
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 0.98rem;
        color: #A0AEC0;
        margin-bottom: 1.4rem;
    }
    div[data-testid="stMetric"] {
        background-color: #161B22;
        border: 1px solid #30363D;
        padding: 15px 20px;
        border-radius: 10px;
    }
    .sidebar-credit {
        background-color: #161B22;
        border: 1px solid rgba(0, 212, 255, 0.4);
        padding: 12px 14px;
        border-radius: 10px;
        margin-top: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Cached Data Loaders
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="Initializing SQLite database...")
def load_base_datasets() -> tuple[str, pd.DataFrame, pd.DataFrame]:
    status_msg = initialize_database(force_reload=False)
    matches = run_query("SELECT * FROM matches;")
    deliveries = run_query("SELECT * FROM deliveries;")
    return status_msg, matches, deliveries


@st.cache_data(show_spinner="Running SQL query...")
def execute_cached_sql(query: str) -> pd.DataFrame:
    return run_query(query)


def apply_plotly_theme(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=45, b=20),
        font=dict(color="#E2E8F0"),
    )
    return fig


# -----------------------------------------------------------------------------
# Main Application
# -----------------------------------------------------------------------------
def main():
    st.markdown(
        '<div class="main-header">🏏 CricInsights: T20 & League Match Analytics Engine</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="author-badge">✨ Made by Uday Singh &nbsp;|&nbsp; Python Project</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">End-to-End SQLite, Pandas & Plotly Sports Intelligence Dashboard</div>',
        unsafe_allow_html=True,
    )

    db_status, matches_df, deliveries_df = load_base_datasets()

    # -------------------------------------------------------------------------
    # Left Sidebar Filters
    # -------------------------------------------------------------------------
    st.sidebar.header("🎛️ Analytics Filters")
    st.sidebar.caption(f"🗄️ {db_status}")

    if st.sidebar.button("🔄 Reload Database from CSV", use_container_width=True):
        initialize_database(force_reload=True)
        st.cache_data.clear()
        st.rerun()

    st.sidebar.divider()

    available_seasons = sorted(matches_df["season"].unique().tolist(), reverse=True)
    selected_seasons = st.sidebar.multiselect(
        "Select Season(s)",
        options=available_seasons,
        default=[],
        help="Leave empty to include all seasons.",
    )

    all_teams = sorted(
        list(set(matches_df["team1"].unique()).union(set(matches_df["team2"].unique())))
    )
    all_teams = [t for t in all_teams if t != "No Result"]
    selected_teams = st.sidebar.multiselect(
        "Select Team(s)",
        options=all_teams,
        default=[],
        help="Filters matches where at least one selected team participated.",
    )

    selected_toss = st.sidebar.selectbox(
        "Toss Decision",
        options=["All", "bat", "field"],
        index=0,
    )

    st.sidebar.markdown(
        """
        <div class="sidebar-credit">
            <div style="color:#00D4FF; font-weight:700; font-size:0.92rem;">👨‍💻 Made by Uday Singh</div>
            <div style="color:#E2E8F0; font-size:0.80rem; margin-top:3px;">Python Project • MSc Computer Science</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Apply Filters
    filtered_matches = matches_df.copy()

    if selected_seasons:
        filtered_matches = filtered_matches[
            filtered_matches["season"].isin(selected_seasons)
        ]

    if selected_teams:
        filtered_matches = filtered_matches[
            filtered_matches["team1"].isin(selected_teams)
            | filtered_matches["team2"].isin(selected_teams)
        ]

    if selected_toss != "All":
        filtered_matches = filtered_matches[
            filtered_matches["toss_decision"] == selected_toss
        ]

    valid_match_ids = set(filtered_matches["id"].unique())
    filtered_deliveries = deliveries_df[
        deliveries_df["match_id"].isin(valid_match_ids)
    ].copy()

    # -------------------------------------------------------------------------
    # Top Row: 4 KPI Cards
    # -------------------------------------------------------------------------
    kpis = get_kpi_metrics(filtered_matches, filtered_deliveries)

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Matches", f"{kpis['total_matches']:,}")
    kpi2.metric("Total Runs Scored", f"{kpis['total_runs']:,}")
    kpi3.metric("Toss-Win Match-Win %", f"{kpis['toss_win_match_win_pct']:.1f}%")
    kpi4.metric("Total Wickets Fallen", f"{kpis['total_wickets']:,}")

    st.divider()

    if filtered_matches.empty:
        st.warning("No matches match the selected filters. Adjust your sidebar filters.")
        return

    # -------------------------------------------------------------------------
    # 4 Organized Analytical Tabs
    # -------------------------------------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📊 Match & Toss Insights",
            "🏏 Batting & Bowling Stats",
            "⚔️ Head-to-Head & Venues",
            "🗄️ SQLite Explorer",
        ]
    )

    # TAB 1: Match & Toss Insights
    with tab1:
        st.subheader("Team Performance & Toss Strategy Impact")

        team_perf_df = get_team_performance(filtered_matches)
        toss_dist_df, toss_rate_df = get_toss_impact(filtered_matches)

        col1, col2 = st.columns([1.4, 1])

        with col1:
            fig_team = px.bar(
                team_perf_df,
                x="Team",
                y="Wins",
                color="Win_Pct",
                text="Wins",
                color_continuous_scale="Tealgrn",
                hover_data=["Matches", "Losses_NR", "Win_Pct"],
                title="Matches Won & Win Percentage by Team",
            )
            fig_team.update_traces(textposition="outside")
            fig_team.update_layout(xaxis_tickangle=-30)
            st.plotly_chart(apply_plotly_theme(fig_team), use_container_width=True)

        with col2:
            fig_toss_pie = px.pie(
                toss_dist_df,
                names="Decision",
                values="Count",
                hole=0.45,
                color="Decision",
                color_discrete_map={"Field": "#00D4FF", "Bat": "#FF6B6B"},
                title="Toss Decision Distribution (Bat vs Field)",
            )
            fig_toss_pie.update_traces(textinfo="percent+label")
            st.plotly_chart(apply_plotly_theme(fig_toss_pie), use_container_width=True)

        col3, col4 = st.columns([1, 1.4])

        with col3:
            if not toss_rate_df.empty:
                fig_toss_win = px.bar(
                    toss_rate_df,
                    x="Decision",
                    y="Win_Rate",
                    color="Decision",
                    text="Win_Rate",
                    color_discrete_map={"Field": "#00D4FF", "Bat": "#FF6B6B"},
                    hover_data=["Total_Tosses", "Matches_Won"],
                    title="Match Win Rate (%) After Winning Toss",
                )
                fig_toss_win.update_traces(
                    texttemplate="%{text:.1f}%", textposition="outside"
                )
                fig_toss_win.update_yaxes(range=[0, 100])
                st.plotly_chart(apply_plotly_theme(fig_toss_win), use_container_width=True)

        with col4:
            st.markdown("#### Team Standings Summary")
            st.dataframe(
                team_perf_df.style.format({"Win_Pct": "{:.2f}%"}),
                use_container_width=True,
                hide_index=True,
            )

    # TAB 2: Batting & Bowling Stats
    with tab2:
        st.subheader("Top 10 Run Scorers & Wicket Takers")

        top_batters_df = get_top_run_scorers(filtered_deliveries, top_n=10)
        top_bowlers_df = get_top_wicket_takers(filtered_deliveries, top_n=10)

        bat_col, bowl_col = st.columns(2)

        with bat_col:
            st.markdown("#### 🏏 Top 10 Run Scorers")
            if not top_batters_df.empty:
                fig_bat = px.bar(
                    top_batters_df.sort_values("Total_Runs", ascending=True),
                    x="Total_Runs",
                    y="Batter",
                    orientation="h",
                    color="Strike_Rate",
                    text="Total_Runs",
                    color_continuous_scale="Blues",
                    hover_data=["Balls_Faced", "Fours", "Sixes", "Strike_Rate"],
                    title="Runs Scored (Colored by Strike Rate)",
                )
                fig_bat.update_traces(textposition="outside")
                st.plotly_chart(apply_plotly_theme(fig_bat), use_container_width=True)

                st.dataframe(
                    top_batters_df.style.format({"Strike_Rate": "{:.2f}"}),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No batting data available for the selected filters.")

        with bowl_col:
            st.markdown("#### 🎯 Top 10 Wicket Takers")
            if not top_bowlers_df.empty:
                fig_bowl = px.bar(
                    top_bowlers_df.sort_values("Wickets", ascending=True),
                    x="Wickets",
                    y="Bowler",
                    orientation="h",
                    color="Economy_Rate",
                    text="Wickets",
                    color_continuous_scale="Sunsetdark_r",
                    hover_data=["Overs", "Runs_Conceded", "Economy_Rate"],
                    title="Wickets Taken (Colored by Economy Rate)",
                )
                fig_bowl.update_traces(textposition="outside")
                st.plotly_chart(apply_plotly_theme(fig_bowl), use_container_width=True)

                st.dataframe(
                    top_bowlers_df.style.format(
                        {"Overs": "{:.1f}", "Economy_Rate": "{:.2f}"}
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No bowling data available for the selected filters.")

    # TAB 3: Head-to-Head & Venues
    with tab3:
        st.subheader("⚔️ Head-to-Head Rivalry Comparison")

        if len(all_teams) >= 2:
            sel_col1, sel_col2 = st.columns(2)
            with sel_col1:
                team_a = st.selectbox("Select Team A", options=all_teams, index=0)
            with sel_col2:
                default_b = 1 if len(all_teams) > 1 else 0
                team_b = st.selectbox("Select Team B", options=all_teams, index=default_b)

            if team_a == team_b:
                st.warning("Please select two different teams to compare.")
            else:
                h2h = get_head_to_head(filtered_matches, team_a, team_b)
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Encounters", h2h["total_matches"])
                m2.metric(f"{team_a} Wins", h2h["team_a_wins"])
                m3.metric(f"{team_b} Wins", h2h["team_b_wins"])
                m4.metric("No Result / Tie", h2h["no_result"])

                if h2h["total_matches"] > 0:
                    fig_h2h = px.bar(
                        h2h["summary_df"],
                        x="Outcome",
                        y="Matches",
                        color="Outcome",
                        text="Matches",
                        color_discrete_sequence=["#00D4FF", "#FF6B6B", "#A0AEC0"],
                        title=f"Head-to-Head: {team_a} vs {team_b}",
                    )
                    fig_h2h.update_traces(textposition="outside")
                    st.plotly_chart(apply_plotly_theme(fig_h2h), use_container_width=True)
                else:
                    st.info(
                        f"No matches recorded between {team_a} and {team_b} under current filters."
                    )

        st.divider()
        st.subheader("🏟️ Top High-Scoring Venues")

        venues_df = get_top_venues(filtered_matches, filtered_deliveries, top_n=10)
        if not venues_df.empty:
            v_col1, v_col2 = st.columns([1.4, 1])
            with v_col1:
                fig_venue = px.bar(
                    venues_df.sort_values("Avg_Runs_Per_Match", ascending=True),
                    x="Avg_Runs_Per_Match",
                    y="Venue",
                    orientation="h",
                    color="Matches_Hosted",
                    text="Avg_Runs_Per_Match",
                    color_continuous_scale="Viridis",
                    hover_data=["Matches_Hosted", "Total_Runs"],
                    title="Top Venues by Average Runs Per Match",
                )
                fig_venue.update_traces(
                    texttemplate="%{text:.1f}", textposition="outside"
                )
                st.plotly_chart(apply_plotly_theme(fig_venue), use_container_width=True)

            with v_col2:
                st.dataframe(
                    venues_df.style.format({"Avg_Runs_Per_Match": "{:.2f}"}),
                    use_container_width=True,
                    hide_index=True,
                )
        else:
            st.info("No venue scoring data available for the selected filters.")

    # TAB 4: SQLite Explorer
    with tab4:
        st.subheader("🗄️ Live SQLite Database Explorer (`cricket.db`)")
        st.markdown(
            "Inspect pre-built analytical SQL queries executed directly against the normalized SQLite tables (`matches` and `deliveries`)."
        )

        predefined_queries = {
            "1. Season-Wise Scoring & Boundary Summary (JOIN + GROUP BY)": """
SELECT 
    m.season AS Season,
    COUNT(DISTINCT m.id) AS Matches_Played,
    SUM(d.total_runs) AS Total_Runs,
    SUM(CASE WHEN d.batsman_runs = 4 THEN 1 ELSE 0 END) AS Total_Fours,
    SUM(CASE WHEN d.batsman_runs = 6 THEN 1 ELSE 0 END) AS Total_Sixes,
    ROUND(CAST(SUM(d.total_runs) AS FLOAT) / COUNT(DISTINCT m.id), 2) AS Avg_Runs_Per_Match
FROM matches m
JOIN deliveries d ON m.id = d.match_id
GROUP BY m.season
ORDER BY m.season DESC;
            """.strip(),
            "2. Most Successful Chasing Teams (Toss = Field & Won Match)": """
SELECT 
    winner AS Team,
    COUNT(*) AS Successful_Chases
FROM matches
WHERE toss_decision = 'field' 
  AND toss_winner = winner
  AND winner != 'No Result'
GROUP BY winner
ORDER BY Successful_Chases DESC
LIMIT 10;
            """.strip(),
            "3. Death-Overs Batting Powerhouses (Strike Rate with Min 50 Balls)": """
SELECT 
    batter AS Batter,
    SUM(batsman_runs) AS Runs,
    SUM(CASE WHEN is_wide = 0 THEN 1 ELSE 0 END) AS Balls_Faced,
    ROUND((CAST(SUM(batsman_runs) AS FLOAT) / NULLIF(SUM(CASE WHEN is_wide = 0 THEN 1 ELSE 0 END), 0)) * 100, 2) AS Strike_Rate
FROM deliveries
GROUP BY batter
HAVING Balls_Faced >= 50
ORDER BY Strike_Rate DESC
LIMIT 10;
            """.strip(),
            "4. Most Economical Bowlers (Min 10 Overs Bowled)": """
SELECT 
    bowler AS Bowler,
    SUM(CASE WHEN is_wicket = 1 AND dismissal_kind NOT IN ('run out', 'retired hurt', 'obstructing the field', 'none') THEN 1 ELSE 0 END) AS Wickets,
    ROUND(CAST(SUM(CASE WHEN is_wide = 0 AND is_noball = 0 THEN 1 ELSE 0 END) AS FLOAT) / 6.0, 1) AS Overs,
    ROUND(CAST(SUM(CASE WHEN is_bye_legbye = 1 THEN 0 ELSE total_runs END) AS FLOAT) / (SUM(CASE WHEN is_wide = 0 AND is_noball = 0 THEN 1 ELSE 0 END) / 6.0), 2) AS Economy_Rate
FROM deliveries
GROUP BY bowler
HAVING SUM(CASE WHEN is_wide = 0 AND is_noball = 0 THEN 1 ELSE 0 END) >= 60
ORDER BY Economy_Rate ASC
LIMIT 10;
            """.strip(),
        }

        selected_query_label = st.selectbox(
            "Choose an Analytical SQL Query:",
            options=list(predefined_queries.keys()),
        )
        active_sql = predefined_queries[selected_query_label]

        st.markdown("**Executed SQL Statement:**")
        st.code(active_sql, language="sql")

        st.markdown("**Live Query Result from `cricket.db`:**")
        sql_result_df = execute_cached_sql(active_sql)
        st.dataframe(sql_result_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()