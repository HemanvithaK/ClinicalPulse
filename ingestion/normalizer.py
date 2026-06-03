from pydantic import BaseModel, field_validator
from typing import Optional
from datetime import date


class TrialRecord(BaseModel):
    nct_id: str
    title: str
    status: str
    phase: Optional[str]
    conditions: list[str]
    interventions: list[str]
    sponsor: Optional[str]
    enrollment: Optional[int]
    start_date: Optional[str]
    completion_date: Optional[str]
    last_updated: Optional[str]
    locations: list[str]

    @field_validator("phase", mode="before")
    def clean_phase(cls, v):
        if not v:
            return None
        return v.replace("PHASE", "Phase ").strip()

    @field_validator("enrollment", mode="before")
    def clean_enrollment(cls, v):
        if not v:
            return None
        try:
            return int(v)
        except (ValueError, TypeError):
            return None


def parse_pipe_separated(value: str) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split("|") if item.strip()]


def normalize_trial(raw: dict) -> Optional[TrialRecord]:
    try:
        return TrialRecord(
            nct_id=raw.get("NCT Number", "").strip(),
            title=raw.get("Study Title", "").strip(),
            status=raw.get("Study Status", "").strip(),
            phase=raw.get("Phases", "").strip() or None,
            conditions=parse_pipe_separated(raw.get("Conditions", "")),
            interventions=parse_pipe_separated(raw.get("Interventions", "")),
            sponsor=raw.get("Sponsor", "").strip() or None,
            enrollment=raw.get("Enrollment"),
            start_date=raw.get("Start Date") or None,
            completion_date=raw.get("Primary Completion Date") or None,
            last_updated=raw.get("Last Update Posted") or None,
            locations=parse_pipe_separated(raw.get("Locations", "")),
        )
    except Exception as e:
        print(f"  Skipping trial {raw.get('NCT Number', 'unknown')}: {e}")
        return None


def normalize_all(raw_trials: list[dict]) -> list[TrialRecord]:
    records = []
    skipped = 0
    for raw in raw_trials:
        record = normalize_trial(raw)
        if record:
            records.append(record)
        else:
            skipped += 1
    print(f"Normalized {len(records)} trials, skipped {skipped}")
    return records


if __name__ == "__main__":
    from ingestion.api_client import run_ingestion

    raw = run_ingestion(["cancer"], max_per_condition=10)
    records = normalize_all(raw)

    if records:
        print("\nSample normalized trial:")
        r = records[0]
        print(f"  NCT ID     : {r.nct_id}")
        print(f"  Title      : {r.title}")
        print(f"  Status     : {r.status}")
        print(f"  Phase      : {r.phase}")
        print(f"  Conditions : {r.conditions}")
        print(f"  Enrollment : {r.enrollment}")
        print(f"  Locations  : {r.locations[:2]}")