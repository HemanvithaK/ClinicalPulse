import json
import pathlib
from datetime import datetime
from pytrials.client import ClinicalTrials

STATE_FILE = pathlib.Path("ingestion_state.json")
ct = ClinicalTrials()


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"last_run": "2024-01-01", "ingested_count": 0}


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2))


def fetch_trials(condition: str, max_trials: int = 100) -> list[dict]:
    print(f"  Fetching trials for: {condition}")
    fields = [
        "NCT Number",
        "Study Title",
        "Study Status",
        "Phases",
        "Conditions",
        "Interventions",
        "Enrollment",
        "Start Date",
        "Primary Completion Date",
        "Last Update Posted",
        "Sponsor",
        "Locations",
    ]
    try:
        studies = ct.get_study_fields(
            search_expr=condition,
            fields=fields,
            max_studies=max_trials,
            fmt="csv",
        )
        headers = studies[0]
        data_rows = studies[1:]
        trials = [dict(zip(headers, row)) for row in data_rows]
        print(f"  Got {len(trials)} trials")
        return trials
    except Exception as e:
        # retry with smaller batch if field limit hit
        if "field larger than field limit" in str(e) and max_trials > 100:
            print(f"  Retrying with smaller batch...")
            return fetch_trials(condition, max_trials=100)
        print(f"  Failed: {e}")
        return []


def run_ingestion(conditions: list[str], max_per_condition: int = 100) -> list[dict]:
    state = load_state()
    print(f"Starting ingestion (last run: {state['last_run']})")

    all_trials = []
    for condition in conditions:
        trials = fetch_trials(condition, max_trials=max_per_condition)
        all_trials.extend(trials)

    state["last_run"] = datetime.now().strftime("%Y-%m-%d")
    state["ingested_count"] += len(all_trials)
    save_state(state)

    print(f"\nIngestion complete — {len(all_trials)} total trials")
    return all_trials


if __name__ == "__main__":
    CONDITIONS = [
    "cancer", "breast cancer", "lung cancer", "leukemia", "prostate cancer",
    "diabetes", "type 2 diabetes", "insulin resistance",
    "alzheimer", "dementia", "cognitive impairment",
    "parkinson", "multiple sclerosis", "epilepsy", "schizophrenia",
    "heart disease", "hypertension", "stroke", "atrial fibrillation",
    "depression", "anxiety", "bipolar disorder", "PTSD",
    "HIV", "hepatitis", "tuberculosis", "COVID-19",
    "obesity", "asthma", "arthritis", "kidney disease",
    "liver disease", "osteoporosis", "anemia", "fibromyalgia",
    "celiac disease", "crohn disease", "lupus", "sarcoidosis",
    ]
    trials = run_ingestion(CONDITIONS, max_per_condition=200)

    if trials:
        print("\nSample trial:")
        for k, v in trials[0].items():
            print(f"  {k}: {v}")