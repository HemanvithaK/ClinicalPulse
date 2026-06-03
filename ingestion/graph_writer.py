from neo4j import GraphDatabase
from dotenv import load_dotenv
from ingestion.normalizer import TrialRecord
import os

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "clinical123")


class GraphWriter:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
        )

    def close(self):
        self.driver.close()

    def create_indexes(self):
        with self.driver.session() as session:
            session.run("CREATE INDEX trial_id IF NOT EXISTS FOR (t:Trial) ON (t.nct_id)")
            session.run("CREATE INDEX condition_name IF NOT EXISTS FOR (c:Condition) ON (c.name)")
            session.run("CREATE INDEX drug_name IF NOT EXISTS FOR (d:Drug) ON (d.name)")
            session.run("CREATE INDEX sponsor_name IF NOT EXISTS FOR (s:Sponsor) ON (s.name)")
        print("Indexes created")

    def write_trial(self, trial: TrialRecord):
        with self.driver.session() as session:
            session.execute_write(self._write_trial_tx, trial)

    @staticmethod
    def _write_trial_tx(tx, trial: TrialRecord):
        # create or update the Trial node
        tx.run("""
            MERGE (t:Trial {nct_id: $nct_id})
            SET t.title = $title,
                t.status = $status,
                t.phase = $phase,
                t.enrollment = $enrollment,
                t.start_date = $start_date,
                t.completion_date = $completion_date,
                t.last_updated = $last_updated
        """, 
            nct_id=trial.nct_id,
            title=trial.title,
            status=trial.status,
            phase=trial.phase,
            enrollment=trial.enrollment,
            start_date=trial.start_date,
            completion_date=trial.completion_date,
            last_updated=trial.last_updated,
        )

        # create Condition nodes + TREATS relationship
        for condition in trial.conditions:
            tx.run("""
                MERGE (c:Condition {name: $name})
                WITH c
                MATCH (t:Trial {nct_id: $nct_id})
                MERGE (t)-[:TREATS]->(c)
            """, name=condition, nct_id=trial.nct_id)

        # create Drug nodes + USES_DRUG relationship
        for intervention in trial.interventions:
            # strip "DRUG: " prefix if present
            name = intervention.replace("DRUG: ", "").replace("PROCEDURE: ", "").replace("RADIATION: ", "").strip()
            if not name:
                continue
            tx.run("""
                MERGE (d:Drug {name: $name})
                WITH d
                MATCH (t:Trial {nct_id: $nct_id})
                MERGE (t)-[:USES_DRUG]->(d)
            """, name=name, nct_id=trial.nct_id)

        # create Sponsor node + SPONSORED_BY relationship
        if trial.sponsor:
            tx.run("""
                MERGE (s:Sponsor {name: $name})
                WITH s
                MATCH (t:Trial {nct_id: $nct_id})
                MERGE (t)-[:SPONSORED_BY]->(s)
            """, name=trial.sponsor, nct_id=trial.nct_id)

        # create Location nodes + LOCATED_AT relationship
        for location in trial.locations[:5]:  # cap at 5 per trial
            tx.run("""
                MERGE (l:Location {name: $name})
                WITH l
                MATCH (t:Trial {nct_id: $nct_id})
                MERGE (t)-[:LOCATED_AT]->(l)
            """, name=location, nct_id=trial.nct_id)


def write_all_trials(trials: list[TrialRecord]):
    writer = GraphWriter()
    writer.create_indexes()

    success = 0
    failed = 0

    for trial in trials:
        try:
            writer.write_trial(trial)
            success += 1
        except Exception as e:
            print(f"  Failed to write {trial.nct_id}: {e}")
            failed += 1

    writer.close()
    print(f"Graph write complete — {success} written, {failed} failed")


if __name__ == "__main__":
    from ingestion.api_client import run_ingestion
    from ingestion.normalizer import normalize_all

    print("Fetching trials...")
    raw = run_ingestion(["cancer", "diabetes", "alzheimer"], max_per_condition=50)

    print("Normalizing...")
    records = normalize_all(raw)

    print("Writing to Neo4j...")
    write_all_trials(records)

    print("\nDone! Open http://localhost:7474 and run:")
    print("  MATCH (t:Trial)-[:TREATS]->(c:Condition) RETURN t,c LIMIT 25")