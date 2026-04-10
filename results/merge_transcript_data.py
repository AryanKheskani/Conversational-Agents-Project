import pandas as pd
from pathlib import Path

RESULTS_PATH = Path("Results.csv")
TRANSCRIPTS_PATH = Path("transcript_sessions.csv")
OUTPUT_PATH = Path("Results_with_session_id.csv")

AMSTERDAM_TZ = "Europe/Amsterdam"


def load_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Parse local survey timestamps (day-first), then localize to Amsterdam, then convert to UTC
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

    df["survey_start_dt"] = start_local.dt.tz_localize(AMSTERDAM_TZ).dt.tz_convert("UTC")
    df["survey_end_dt"] = end_local.dt.tz_localize(AMSTERDAM_TZ).dt.tz_convert("UTC")

    return df


def load_transcripts(path: Path) -> pd.DataFrame:
    tx = pd.read_csv(path)

    # Transcript timestamps already include timezone offset, so parse directly to UTC
    tx["timestamp_local_dt"] = pd.to_datetime(
        tx["timestamp_local"],
        errors="coerce",
        utc=True,
    )

    # Normalize use_prosody in case CSV stores it as strings
    tx["use_prosody"] = tx["use_prosody"].map(
        lambda x: True if str(x).strip().lower() == "true"
        else False if str(x).strip().lower() == "false"
        else x
    )

    return tx


def expected_use_prosody(condition_order: str):
    if condition_order == "Prosodic":
        return True
    if condition_order == "Semantic Only":
        return False
    return None


def main() -> None:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"Could not find {RESULTS_PATH.resolve()}")

    if not TRANSCRIPTS_PATH.exists():
        raise FileNotFoundError(f"Could not find {TRANSCRIPTS_PATH.resolve()}")

    results_df = load_results(RESULTS_PATH)
    transcripts_df = load_transcripts(TRANSCRIPTS_PATH)

    matched_session_ids = []

    for _, survey_row in results_df.iterrows():
        start_dt = survey_row["survey_start_dt"]
        end_dt = survey_row["survey_end_dt"]
        condition_order = survey_row["Condition Order"]
        expected_flag = expected_use_prosody(condition_order)

        if pd.isna(start_dt) or pd.isna(end_dt) or expected_flag is None:
            matched_session_ids.append("NA")
            continue

        matches = transcripts_df[
            (transcripts_df["timestamp_local_dt"].notna()) &
            (transcripts_df["timestamp_local_dt"] >= start_dt) &
            (transcripts_df["timestamp_local_dt"] <= end_dt) &
            (transcripts_df["use_prosody"] == expected_flag)
        ].copy()

        matches = matches.sort_values("timestamp_local_dt")

        if matches.empty:
            matched_session_ids.append("NA")
        else:
            matched_session_ids.append(str(matches.iloc[0]["session_id_raw"]))

    output_df = results_df.copy()
    output_df.insert(0, "session_id", matched_session_ids)
    output_df = output_df.drop(columns=["survey_start_dt", "survey_end_dt"])
    output_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved merged file to: {OUTPUT_PATH}")
    print(f"Rows with no match: {(pd.Series(matched_session_ids) == 'NA').sum()}")
    print(f"Rows with a match: {(pd.Series(matched_session_ids) != 'NA').sum()}")


if __name__ == "__main__":
    main()