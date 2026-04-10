import pandas as pd
from pathlib import Path

RESULTS_PATH = Path("Results.csv")
TRANSCRIPTS_PATH = Path("transcript_sessions.csv")
OUTPUT_PATH = Path("Results_with_session_ids.csv")

AMSTERDAM_TZ = "Europe/Amsterdam"


def condition_to_use_prosody(condition_value):
    if pd.isna(condition_value):
        return None

    value = str(condition_value).strip()
    if value == "Prosodic":
        return True
    if value == "Semantic Only":
        return False
    return None


def load_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Survey timestamps are local Amsterdam time in day-first format like 27-3-2026 13:07
    start_local = pd.to_datetime(
        df["Start time"],
        errors="coerce",
        dayfirst=True,
    )
    end_local = pd.to_datetime(
        df["Completion time"],
        errors="coerce",
        dayfirst=True,
    )

    # Make them timezone-aware and comparable with transcript timestamps
    df["survey_start_dt"] = start_local.dt.tz_localize(AMSTERDAM_TZ).dt.tz_convert("UTC")
    df["survey_end_dt"] = end_local.dt.tz_localize(AMSTERDAM_TZ).dt.tz_convert("UTC")

    return df


def load_transcripts(path: Path) -> pd.DataFrame:
    tx = pd.read_csv(path)

    # transcript_sessions.csv timestamp_local already contains timezone offset
    tx["timestamp_local_dt"] = pd.to_datetime(
        tx["timestamp_local"],
        errors="coerce",
        utc=True,
    )

    # Normalize booleans in case they were read as strings
    tx["use_prosody"] = tx["use_prosody"].map(
        lambda x: True if str(x).strip().lower() == "true"
        else False if str(x).strip().lower() == "false"
        else x
    )

    return tx


def first_match_session_id(transcripts_df, start_dt, end_dt, expected_use_prosody):
    if pd.isna(start_dt) or pd.isna(end_dt) or expected_use_prosody is None:
        return "NA"

    matches = transcripts_df[
        (transcripts_df["timestamp_local_dt"].notna()) &
        (transcripts_df["timestamp_local_dt"] >= start_dt) &
        (transcripts_df["timestamp_local_dt"] <= end_dt) &
        (transcripts_df["use_prosody"] == expected_use_prosody)
    ].copy()

    matches = matches.sort_values("timestamp_local_dt")

    if matches.empty:
        return "NA"

    return str(matches.iloc[0]["session_id_raw"])


def insert_before_column(df, new_col_name, values, before_col):
    insert_loc = df.columns.get_loc(before_col)
    df.insert(insert_loc, new_col_name, values)


def main():
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"Could not find {RESULTS_PATH.resolve()}")

    if not TRANSCRIPTS_PATH.exists():
        raise FileNotFoundError(f"Could not find {TRANSCRIPTS_PATH.resolve()}")

    results_df = load_results(RESULTS_PATH)
    transcripts_df = load_transcripts(TRANSCRIPTS_PATH)

    session_ids_a = []
    session_ids_b = []

    for _, row in results_df.iterrows():
        start_dt = row["survey_start_dt"]
        end_dt = row["survey_end_dt"]

        expected_a = condition_to_use_prosody(row.get("Condition Order"))
        expected_b = condition_to_use_prosody(row.get("Condition Order1"))

        session_a = first_match_session_id(transcripts_df, start_dt, end_dt, expected_a)
        session_b = first_match_session_id(transcripts_df, start_dt, end_dt, expected_b)

        session_ids_a.append(session_a)
        session_ids_b.append(session_b)

    output_df = results_df.copy()

    # Insert new columns right before the corresponding condition columns
    insert_before_column(output_df, "session_id_A", session_ids_a, "Condition Order")
    insert_before_column(output_df, "session_id_B", session_ids_b, "Condition Order1")

    # Remove helper datetime columns
    output_df = output_df.drop(columns=["survey_start_dt", "survey_end_dt"])

    output_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved merged file to: {OUTPUT_PATH}")
    print(f"session_id_A matched rows: {(pd.Series(session_ids_a) != 'NA').sum()}")
    print(f"session_id_A unmatched rows: {(pd.Series(session_ids_a) == 'NA').sum()}")
    print(f"session_id_B matched rows: {(pd.Series(session_ids_b) != 'NA').sum()}")
    print(f"session_id_B unmatched rows: {(pd.Series(session_ids_b) == 'NA').sum()}")


if __name__ == "__main__":
    main()