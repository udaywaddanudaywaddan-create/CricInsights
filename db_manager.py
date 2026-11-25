"""
db_manager.py
Data Access & ETL Layer for CricInsights
Made by Uday Singh | Python Project
"""

import os
import random
import sqlite3
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "cricket.db")
DATA_DIR = os.path.join(BASE_DIR, "data")

# Normalize historical franchise rebrands for consistent analytics
TEAM_NAME_MAP = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
    "Deccan Chargers": "Sunrisers Hyderabad",
}


def _find_csv_file(keyword: str) -> str | None:
    """
    Searches both 'data/' folder and the main project folder for CSV files
    matching keyword (e.g., 'match' or 'deliver'), regardless of uppercase/lowercase.
    """
    for folder in [DATA_DIR, BASE_DIR]:
        if os.path.isdir(folder):
            for fname in os.listdir(folder):
                if fname.lower().endswith(".csv") and keyword in fname.lower():
                    return os.path.join(folder, fname)
    return None


def _normalize_team_series(series: pd.Series) -> pd.Series:
    """Cleans extra spaces and standardizes team names."""
    cleaned = series.fillna("Unknown").astype(str).str.strip()
    return cleaned.replace(TEAM_NAME_MAP)


def _clean_matches_df(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans and standardizes columns for the matches table."""
    df = df.copy()
    df.columns = [col.strip().lower() for col in df.columns]

    if "id" not in df.columns:
        if "match_id" in df.columns:
            df.rename(columns={"match_id": "id"}, inplace=True)
        else:
            df["id"] = range(1, len(df) + 1)

    required_defaults = {
        "season": "2024",
        "team1": "Team A",
        "team2": "Team B",
        "toss_winner": "Team A",
        "toss_decision": "field",
        "winner": "No Result",
        "venue": "Unknown Venue",
    }
    for col, default_val in required_defaults.items():
        if col not in df.columns:
            df[col] = default_val

    df["id"] = pd.to_numeric(df["id"], errors="coerce").fillna(0).astype(int)
    df["season"] = df["season"].fillna("Unknown").astype(str).str.strip()

    for team_col in ["team1", "team2", "toss_winner", "winner"]:
        df[team_col] = _normalize_team_series(df[team_col])
        df.loc[df[team_col].isin(["", "nan", "None", "Unknown"]), team_col] = "No Result"

    df["toss_decision"] = (
        df["toss_decision"].fillna("field").astype(str).str.strip().str.lower()
    )
    df["venue"] = df["venue"].fillna("Unknown Venue").astype(str).str.strip()

    keep_cols = [
        "id",
        "season",
        "team1",
        "team2",
        "toss_winner",
        "toss_decision",
        "winner",
        "venue",
    ]
    return df[keep_cols]


def _clean_deliveries_df(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans and standardizes ball-by-ball columns across Kaggle dataset formats."""
    df = df.copy()
    df.columns = [col.strip().lower() for col in df.columns]

    # Handle column name variations across Kaggle IPL CSV versions
    if "batter" not in df.columns and "batsman" in df.columns:
        df.rename(columns={"batsman": "batter"}, inplace=True)
    if "batsman_runs" not in df.columns and "batter_runs" in df.columns:
        df.rename(columns={"batter_runs": "batsman_runs"}, inplace=True)

    defaults = {
        "match_id": 1,
        "inning": 1,
        "batting_team": "Unknown",
        "bowling_team": "Unknown",
        "batter": "Unknown Batter",
        "bowler": "Unknown Bowler",
        "batsman_runs": 0,
    }
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val

    if "total_runs" not in df.columns:
        df["total_runs"] = df["batsman_runs"]

    df["match_id"] = pd.to_numeric(df["match_id"], errors="coerce").fillna(0).astype(int)
    df["inning"] = pd.to_numeric(df["inning"], errors="coerce").fillna(1).astype(int)
    df["batting_team"] = _normalize_team_series(df["batting_team"])
    df["bowling_team"] = _normalize_team_series(df["bowling_team"])
    df["batter"] = df["batter"].fillna("Unknown Batter").astype(str).str.strip()
    df["bowler"] = df["bowler"].fillna("Unknown Bowler").astype(str).str.strip()

    df["batsman_runs"] = (
        pd.to_numeric(df["batsman_runs"], errors="coerce").fillna(0).astype(int)
    )
    df["total_runs"] = (
        pd.to_numeric(df["total_runs"], errors="coerce").fillna(0).astype(int)
    )

    if "dismissal_kind" not in df.columns:
        df["dismissal_kind"] = "none"
    df["dismissal_kind"] = (
        df["dismissal_kind"].fillna("none").astype(str).str.strip().str.lower()
    )
    df.loc[df["dismissal_kind"].isin(["", "nan", "null"]), "dismissal_kind"] = "none"

    if "is_wicket" in df.columns:
        df["is_wicket"] = (
            pd.to_numeric(df["is_wicket"], errors="coerce").fillna(0).astype(int)
        )
    elif "player_dismissed" in df.columns:
        df["is_wicket"] = df["player_dismissed"].notna().astype(int)
    else:
        df["is_wicket"] = (df["dismissal_kind"] != "none").astype(int)

    # Identify wides, no-balls, and byes/leg-byes for official Strike Rate & Economy rules
    if "extras_type" in df.columns:
        extras_lower = df["extras_type"].fillna("").astype(str).str.lower()
        df["is_wide"] = (extras_lower == "wides").astype(int)
        df["is_noball"] = (extras_lower == "noballs").astype(int)
        df["is_bye_legbye"] = extras_lower.isin(["byes", "legbyes"]).astype(int)
    else:
        wide_col = pd.to_numeric(df.get("wide_runs", 0), errors="coerce").fillna(0)
        noball_col = pd.to_numeric(df.get("noball_runs", 0), errors="coerce").fillna(0)
        bye_col = pd.to_numeric(df.get("bye_runs", 0), errors="coerce").fillna(0)
        legbye_col = pd.to_numeric(df.get("legbye_runs", 0), errors="coerce").fillna(0)
        df["is_wide"] = (wide_col > 0).astype(int)
        df["is_noball"] = (noball_col > 0).astype(int)
        df["is_bye_legbye"] = ((bye_col + legbye_col) > 0).astype(int)

    keep_cols = [
        "match_id",
        "inning",
        "batting_team",
        "bowling_team",
        "batter",
        "bowler",
        "batsman_runs",
        "total_runs",
        "is_wicket",
        "dismissal_kind",
        "is_wide",
        "is_noball",
        "is_bye_legbye",
    ]
    return df[keep_cols]


def _generate_fallback_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generates sample T20 data if CSV files are missing."""
    rng = random.Random(42)
    rosters = {
        "Mumbai Indians": {
            "batters": ["R Sharma", "S Yadav", "I Kishan", "T Varma", "H Pandya"],
            "bowlers": ["J Bumrah", "G Coetzee", "P Chawla", "A Madhwal", "H Pandya"],
        },
        "Chennai Super Kings": {
            "batters": ["R Gaikwad", "D Conway", "S Dube", "R Jadeja", "MS Dhoni"],
            "bowlers": ["M Pathirana", "D Chahar", "R Jadeja", "M Theekshana", "T Deshpande"],
        },
        "Royal Challengers Bengaluru": {
            "batters": ["V Kohli", "F du Plessis", "G Maxwell", "R Patidar", "D Karthik"],
            "bowlers": ["M Siraj", "Y Dayal", "K Sharma", "G Maxwell", "L Ferguson"],
        },
        "Kolkata Knight Riders": {
            "batters": ["S Narine", "P Salt", "S Iyer", "A Russell", "R Singh"],
            "bowlers": ["V Chakravarthy", "S Narine", "H Rana", "M Starc", "A Russell"],
        },
        "Rajasthan Royals": {
            "batters": ["Y Jaiswal", "J Buttler", "S Samson", "R Parag", "S Hetmyer"],
            "bowlers": ["T Boult", "Y Chahal", "R Ashwin", "S Sharma", "A Khan"],
        },
        "Sunrisers Hyderabad": {
            "batters": ["T Head", "A Sharma", "H Klaasen", "A Markram", "N Reddy"],
            "bowlers": ["P Cummins", "B Kumar", "T Natarajan", "S Ahmed", "J Unadkat"],
        },
    }
    venues = [
        "Wankhede Stadium, Mumbai",
        "M. A. Chidambaram Stadium, Chennai",
        "M. Chinnaswamy Stadium, Bengaluru",
        "Eden Gardens, Kolkata",
        "Sawai Mansingh Stadium, Jaipur",
        "Rajiv Gandhi International Stadium, Hyderabad",
    ]
    teams = list(rosters.keys())
    dismissal_types = ["caught", "bowled", "lbw", "run out", "stumped"]
    matches_records, deliveries_records = [], []
    match_id = 1

    for season in ["2022", "2023", "2024"]:
        for _ in range(20):
            team1, team2 = rng.sample(teams, 2)
            venue = rng.choice(venues)
            toss_winner = rng.choice([team1, team2])
            toss_decision = rng.choice(["bat", "field"])
            if toss_decision == "bat":
                bat_first, bat_second = toss_winner, (team2 if toss_winner == team1 else team1)
            else:
                bat_second, bat_first = toss_winner, (team2 if toss_winner == team1 else team1)

            inning_totals = {}
            for inning_num, (bat_team, bowl_team) in enumerate(
                [(bat_first, bat_second), (bat_second, bat_first)], start=1
            ):
                batters = rosters[bat_team]["batters"]
                bowlers = rosters[bowl_team]["bowlers"]
                wickets_fallen, total_score = 0, 0
                for ball_idx in range(120):
                    if wickets_fallen >= 9:
                        break
                    batter = batters[min(wickets_fallen // 2, len(batters) - 1)]
                    bowler = bowlers[(ball_idx // 6) % len(bowlers)]
                    is_wide = 1 if rng.random() < 0.04 else 0
                    is_noball = 1 if (not is_wide and rng.random() < 0.01) else 0
                    if is_wide or is_noball:
                        batsman_runs, total_runs, is_wicket, dismissal_kind = 0, 1, 0, "none"
                    else:
                        is_wicket = 1 if rng.random() < 0.055 else 0
                        if is_wicket:
                            batsman_runs, total_runs = 0, 0
                            dismissal_kind = rng.choice(dismissal_types)
                            wickets_fallen += 1
                        else:
                            batsman_runs = rng.choices(
                                [0, 1, 2, 3, 4, 6], weights=[32, 36, 9, 2, 14, 7]
                            )[0]
                            total_runs, dismissal_kind = batsman_runs, "none"
                    total_score += total_runs
                    deliveries_records.append(
                        {
                            "match_id": match_id,
                            "inning": inning_num,
                            "batting_team": bat_team,
                            "bowling_team": bowl_team,
                            "batter": batter,
                            "bowler": bowler,
                            "batsman_runs": batsman_runs,
                            "total_runs": total_runs,
                            "is_wicket": is_wicket,
                            "dismissal_kind": dismissal_kind,
                            "is_wide": is_wide,
                            "is_noball": is_noball,
                            "is_bye_legbye": 0,
                        }
                    )
                inning_totals[bat_team] = total_score

            winner = (
                bat_first
                if inning_totals[bat_first] > inning_totals[bat_second]
                else (
                    bat_second
                    if inning_totals[bat_second] > inning_totals[bat_first]
                    else toss_winner
                )
            )
            matches_records.append(
                {
                    "id": match_id,
                    "season": season,
                    "team1": team1,
                    "team2": team2,
                    "toss_winner": toss_winner,
                    "toss_decision": toss_decision,
                    "winner": winner,
                    "venue": venue,
                }
            )
            match_id += 1

    return pd.DataFrame(matches_records), pd.DataFrame(deliveries_records)


def initialize_database(force_reload: bool = False) -> str:
    """Initializes 'cricket.db' from CSV files or fallback generator."""
    matches_path = _find_csv_file("match")
    deliveries_path = _find_csv_file("deliver")

    if not force_reload and os.path.exists(DB_PATH):
        try:
            with sqlite3.connect(DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('matches', 'deliveries');"
                )
                tables = {row[0] for row in cursor.fetchall()}
                if {"matches", "deliveries"}.issubset(tables):
                    m_count = cursor.execute("SELECT COUNT(*) FROM matches;").fetchone()[0]
                    d_count = cursor.execute("SELECT COUNT(*) FROM deliveries;").fetchone()[0]
                    # Automatically replace old 60-match fallback DB if real CSVs exist
                    if m_count > 0 and d_count > 0:
                        if not (matches_path and deliveries_path and m_count == 60):
                            return f"Connected to cricket.db ({m_count:,} matches, {d_count:,} deliveries)"
        except sqlite3.Error:
            pass

    try:
        if not (matches_path and deliveries_path):
            raise FileNotFoundError("CSV files not found.")

        raw_matches = pd.read_csv(matches_path)
        raw_deliveries = pd.read_csv(deliveries_path)

        if raw_matches.empty or raw_deliveries.empty:
            raise ValueError("One or more CSV files are empty.")

        matches_df = _clean_matches_df(raw_matches)
        deliveries_df = _clean_deliveries_df(raw_deliveries)
        source_status = (
            f"Loaded CSV Dataset ({len(matches_df):,} matches, {len(deliveries_df):,} deliveries)"
        )

    except (FileNotFoundError, pd.errors.EmptyDataError, ValueError):
        matches_df, deliveries_df = _generate_fallback_dataset()
        source_status = (
            f"Fallback Mode ({len(matches_df):,} matches, {len(deliveries_df):,} deliveries)"
        )

    with sqlite3.connect(DB_PATH) as conn:
        matches_df.to_sql("matches", conn, if_exists="replace", index=False)
        deliveries_df.to_sql("deliveries", conn, if_exists="replace", index=False)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_matches_id ON matches(id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_matches_season ON matches(season);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_deliveries_match_id ON deliveries(match_id);")

    return source_status


def run_query(sql_query: str) -> pd.DataFrame:
    """Executes a SQL query on 'cricket.db' and returns a Pandas DataFrame."""
    if not os.path.exists(DB_PATH):
        initialize_database()

    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(sql_query, conn)