import pandas as pd
import numpy as np
from scipy import stats
import pingouin as pg
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
import zipfile
import json
from datetime import datetime, timezone
from pathlib import Path
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# X. READ TRANSCRIPTS FROM ZIP AND EXTRACT SESSION TIMES
# ─────────────────────────────────────────────
zip_path = Path('../transcripts.zip')

transcript_rows = []

if zip_path.exists():
    with zipfile.ZipFile(zip_path, 'r') as z:
        for name in z.namelist():
            normalized_name = name.replace('\\', '/')

            # Skip directories
            if normalized_name.endswith('/'):
                continue

            # Skip macOS metadata
            if normalized_name.startswith('__MACOSX/'):
                continue

            # Only keep JSON files inside transcripts/
            if not normalized_name.startswith('transcripts/'):
                continue
            if not normalized_name.endswith('.json'):
                continue

            try:
                with z.open(name) as f:
                    data = json.load(f)

                session_id = data.get('session_id', None)

                if session_id is None:
                    continue  # skip silently

                # Extract numeric part (still needed internally for timestamp)
                session_number = int(str(session_id).replace('session_', ''))

                # Convert timestamp
                dt_utc = datetime.fromtimestamp(session_number, tz=timezone.utc)

                # 🔥 Strongly recommended: fixed timezone
                import zoneinfo
                amsterdam = zoneinfo.ZoneInfo("Europe/Amsterdam")
                dt_local = dt_utc.astimezone(amsterdam)

                print(
                    f"{normalized_name} | "
                    f"{session_id} | "
                    f"Local: {dt_local.strftime('%Y-%m-%d %H:%M:%S')}"
                )

                transcript_rows.append({
                    'zip_member': normalized_name,
                    'session_id_raw': session_id,
                    'timestamp_local': dt_local.isoformat(),
                    'mode': data.get('mode'),
                    'use_prosody': data.get('use_prosody'),
                    'n_turns': len(data.get('turns', []))
                })

            except Exception as e:
                print(f"[ERROR] {normalized_name} → {str(e)}")
                continue

    transcripts_df = pd.DataFrame(transcript_rows)
    transcripts_df = transcripts_df.sort_values('timestamp_local')
    transcripts_df.to_csv('transcript_sessions.csv', index=False)

    print("\n" + "=" * 60)
    print("TRANSCRIPT SESSION EXTRACTION")
    print("=" * 60)
    print(f"Read {len(transcripts_df)} transcript files from {zip_path}")
    print("Saved transcript session metadata to transcript_sessions.csv")

    if not transcripts_df.empty:
        print("\nFirst few extracted sessions:")
        print(transcripts_df[['zip_member', 'session_id_raw', 'timestamp_local']].head().to_string(index=False))

else:
    print(f"\ntranscripts.zip not found at: {zip_path.resolve()}")