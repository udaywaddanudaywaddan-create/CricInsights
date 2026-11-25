"""
analytics.py
Core Analytical Engine for CricInsights using Pandas.

Contains modular, pure-Pandas functions for:
a) KPI Summary Metrics
b) Team Performance & Win Percentage
c) Toss Decision Impact (Bat vs Field)
d) Top 10 Run Scorers (Runs, Balls, Strike Rate)
e) Top 10 Wicket Takers (Wickets, Runs Conceded, Economy Rate)
f) Head-to-Head Team Comparison
g) Top High-Scoring Venues
 """

import pandas as pd

# Dismissals not credited to the bowler
NON_BOWLER_DISMISSALS = {
    "run out",
    "retired hurt",
    "retired out",
    "obstructing the field",
    "none",
}


def get_kpi_metrics(matches_df: pd.DataFrame, deliveries_df: pd.DataFrame) -> dict:
    """
    Computes top-level summary KPIs for the active match slice.
    Returns total matches, total runs, toss-win match-win %, and total wickets.
    """
    if matches_df.empty:
        return {
            "total_matches": 0,
            "total_runs": 0,
            "toss_win_match_win_pct": 0.0,
            "total_wickets": 0,
        }

    total_matches = int(len(matches_df))
    total_runs = int(deliveries_df["total_runs"].sum()) if not deliveries_df.empty else 0
    total_wickets = int(deliveries_df["is_wicket"].sum()) if not deliveries_df.empty else 0

    valid_results = matches_df[matches_df["winner"] != "No Result"]
    if len(valid_results) > 0:
        toss_and_match_wins = (
            valid_results["toss_winner"] == valid_results["winner"]
        ).sum()
        toss_win_pct = round((toss_and_match_wins / len(valid_results)) * 100, 2)
    else:
        toss_win_pct = 0.0

    return {
        "total_matches": total_matches,
        "total_runs": total_runs,
        "toss_win_match_win_pct": toss_win_pct,
        "total_wickets": total_wickets,
    }


def get_team_performance(matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates Matches Played, Wins, Losses/NR, and Win % for every team.
    """
    if matches_df.empty:
        return pd.DataFrame(columns=["Team", "Matches", "Wins", "Losses_NR", "Win_Pct"])

    team1_counts = matches_df["team1"].value_counts()
    team2_counts = matches_df["team2"].value_counts()
    total_played = team1_counts.add(team2_counts, fill_value=0).astype(int)

    valid_winners = matches_df[matches_df["winner"] != "No Result"]["winner"]
    wins = valid_winners.value_counts()

    perf_df = pd.DataFrame({"Matches": total_played})
    perf_df["Wins"] = perf_df.index.map(wins).fillna(0).astype(int)
    perf_df["Losses_NR"] = perf_df["Matches"] - perf_df["Wins"]
    perf_df["Win_Pct"] = perf_df.apply(
        lambda row: round((row["Wins"] / row["Matches"]) * 100, 2)
        if row["Matches"] > 0
        else 0.0,
        axis=1,
    )

    perf_df.index.name = "Team"
    perf_df = perf_df.reset_index()
    return perf_df.sort_values(by=["Wins", "Win_Pct"], ascending=[False, False]).reset_index(
        drop=True
    )


def get_toss_impact(matches_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyzes toss decisions:
    1. Distribution of 'bat' vs 'field' choices.
    2. Match win percentage when toss winner elects to 'bat' vs 'field'.
    """
    if matches_df.empty:
        empty_dist = pd.DataFrame(columns=["Decision", "Count", "Percentage"])
        empty_rate = pd.DataFrame(
            columns=["Decision", "Total_Tosses", "Matches_Won", "Win_Rate"]
        )
        return empty_dist, empty_rate

    df = matches_df.copy()
    df["Decision"] = df["toss_decision"].str.capitalize()

    # 1. Decision Distribution
    dist_df = (
        df["Decision"]
        .value_counts()
        .reset_index()
        .rename(columns={"count": "Count"})
    )
    total_tosses = dist_df["Count"].sum()
    dist_df["Percentage"] = round((dist_df["Count"] / total_tosses) * 100, 2)

    # 2. Win Rate by Toss Decision (excluding No Result matches)
    valid_df = df[df["winner"] != "No Result"].copy()
    if valid_df.empty:
        empty_rate = pd.DataFrame(
            columns=["Decision", "Total_Tosses", "Matches_Won", "Win_Rate"]
        )
        return dist_df, empty_rate

    valid_df["Toss_Winner_Won"] = (
        valid_df["toss_winner"] == valid_df["winner"]
    ).astype(int)

    rate_df = (
        valid_df.groupby("Decision", as_index=False)
        .agg(
            Total_Tosses=("Toss_Winner_Won", "count"),
            Matches_Won=("Toss_Winner_Won", "sum"),
        )
    )
    rate_df["Win_Rate"] = round(
        (rate_df["Matches_Won"] / rate_df["Total_Tosses"]) * 100, 2
    )

    return dist_df, rate_df


def get_top_run_scorers(deliveries_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Computes Top N Run Scorers with Total Runs, Legal Balls Faced, Fours, Sixes,
    and Strike Rate.
    """
    cols = ["Batter", "Total_Runs", "Balls_Faced", "Fours", "Sixes", "Strike_Rate"]
    if deliveries_df.empty:
        return pd.DataFrame(columns=cols)

    df = deliveries_df.copy()
    # Legal balls faced by a batter exclude wides
    df["legal_ball_faced"] = (df["is_wide"] == 0).astype(int)
    df["is_four"] = (df["batsman_runs"] == 4).astype(int)
    df["is_six"] = (df["batsman_runs"] == 6).astype(int)

    batting_stats = (
        df.groupby("batter", as_index=False)
        .agg(
            Total_Runs=("batsman_runs", "sum"),
            Balls_Faced=("legal_ball_faced", "sum"),
            Fours=("is_four", "sum"),
            Sixes=("is_six", "sum"),
        )
        .rename(columns={"batter": "Batter"})
    )

    batting_stats["Strike_Rate"] = batting_stats.apply(
        lambda row: round((row["Total_Runs"] / row["Balls_Faced"]) * 100, 2)
        if row["Balls_Faced"] > 0
        else 0.0,
        axis=1,
    )

    return (
        batting_stats.sort_values(
            by=["Total_Runs", "Strike_Rate"], ascending=[False, False]
        )
        .head(top_n)
        .reset_index(drop=True)
    )


def get_top_wicket_takers(deliveries_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Computes Top N Wicket Takers with Wickets, Overs Bowled, Runs Conceded,
    and Economy Rate.
    """
    cols = ["Bowler", "Wickets", "Overs", "Runs_Conceded", "Economy_Rate"]
    if deliveries_df.empty:
        return pd.DataFrame(columns=cols)

    df = deliveries_df.copy()

    # Bowler-credited wickets exclude run-outs and non-bowler dismissals
    df["bowler_wicket"] = (
        (df["is_wicket"] == 1) & (~df["dismissal_kind"].isin(NON_BOWLER_DISMISSALS))
    ).astype(int)

    # Legal deliveries exclude wides and no-balls
    df["legal_delivery"] = ((df["is_wide"] == 0) & (df["is_noball"] == 0)).astype(int)

    # Bowler runs conceded exclude byes and leg-byes
    df["bowler_runs"] = df.apply(
        lambda row: 0 if row["is_bye_legbye"] == 1 else row["total_runs"], axis=1
    )

    bowling_stats = (
        df.groupby("bowler", as_index=False)
        .agg(
            Wickets=("bowler_wicket", "sum"),
            Legal_Balls=("legal_delivery", "sum"),
            Runs_Conceded=("bowler_runs", "sum"),
        )
        .rename(columns={"bowler": "Bowler"})
    )

    # Filter bowlers who have bowled at least 1 legal over (6 balls)
    bowling_stats = bowling_stats[bowling_stats["Legal_Balls"] >= 6].copy()

    bowling_stats["Overs"] = bowling_stats["Legal_Balls"].apply(
        lambda balls: round((balls // 6) + ((balls % 6) / 10), 1)
    )
    bowling_stats["Economy_Rate"] = bowling_stats.apply(
        lambda row: round(row["Runs_Conceded"] / (row["Legal_Balls"] / 6), 2)
        if row["Legal_Balls"] > 0
        else 0.0,
        axis=1,
    )

    return (
        bowling_stats[cols]
        .sort_values(by=["Wickets", "Economy_Rate"], ascending=[False, True])
        .head(top_n)
        .reset_index(drop=True)
    )


def get_head_to_head(matches_df: pd.DataFrame, team_a: str, team_b: str) -> dict:
    """
    Compares historical performance between two selected teams.
    """
    h2h_matches = matches_df[
        ((matches_df["team1"] == team_a) & (matches_df["team2"] == team_b))
        | ((matches_df["team1"] == team_b) & (matches_df["team2"] == team_a))
    ]

    total = int(len(h2h_matches))
    if total == 0:
        return {
            "total_matches": 0,
            "team_a_wins": 0,
            "team_b_wins": 0,
            "no_result": 0,
            "summary_df": pd.DataFrame(columns=["Outcome", "Matches"]),
        }

    team_a_wins = int((h2h_matches["winner"] == team_a).sum())
    team_b_wins = int((h2h_matches["winner"] == team_b).sum())
    no_result = total - (team_a_wins + team_b_wins)

    summary_df = pd.DataFrame(
        {
            "Outcome": [f"{team_a} Wins", f"{team_b} Wins", "No Result / Tie"],
            "Matches": [team_a_wins, team_b_wins, no_result],
        }
    )

    return {
        "total_matches": total,
        "team_a_wins": team_a_wins,
        "team_b_wins": team_b_wins,
        "no_result": no_result,
        "summary_df": summary_df,
    }


def get_top_venues(
    matches_df: pd.DataFrame, deliveries_df: pd.DataFrame, top_n: int = 10
) -> pd.DataFrame:
    """
    Identifies top high-scoring venues ranked by Average Runs per Match.
    """
    cols = ["Venue", "Matches_Hosted", "Total_Runs", "Avg_Runs_Per_Match"]
    if matches_df.empty or deliveries_df.empty:
        return pd.DataFrame(columns=cols)

    match_runs = (
        deliveries_df.groupby("match_id", as_index=False)["total_runs"]
        .sum()
        .rename(columns={"match_id": "id", "total_runs": "Match_Runs"})
    )

    merged = pd.merge(matches_df[["id", "venue"]], match_runs, on="id", how="inner")
    if merged.empty:
        return pd.DataFrame(columns=cols)

    venue_stats = (
        merged.groupby("venue", as_index=False)
        .agg(
            Matches_Hosted=("id", "count"),
            Total_Runs=("Match_Runs", "sum"),
        )
        .rename(columns={"venue": "Venue"})
    )

    venue_stats["Avg_Runs_Per_Match"] = round(
        venue_stats["Total_Runs"] / venue_stats["Matches_Hosted"], 2
    )

    return (
        venue_stats.sort_values(
            by=["Avg_Runs_Per_Match", "Total_Runs"], ascending=[False, False]
        )
        .head(top_n)
        .reset_index(drop=True)
    )