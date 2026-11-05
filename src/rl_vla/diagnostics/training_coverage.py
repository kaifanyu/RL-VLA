"""Count finalized training-episode resets without implying exhaustive state visits."""

from collections import Counter
import hashlib
import json
from pathlib import Path


def final_training_reset_coverage(root, tasks=("object_0", "object_1", "object_2")):
    root = Path(root)
    summaries, visits = [], []
    for task in tasks:
        directory = root / "methods/rldt" / task / "training"
        final_path, episodes_path = directory / "summary.json", directory / "episodes.jsonl"
        record = {"task": task, "status": "pending_final_training_summary", "prescribed_training_states": 100}
        if not final_path.exists():
            summaries.append(record)
            continue
        summary_bytes = final_path.read_bytes()
        summary = json.loads(summary_bytes)
        if summary["iterations"] != 100:
            raise ValueError(f"Final training iterations differ from requested100: {task}")
        protocol_path = root / "tasks" / task / "protocol.json"
        protocol_bytes = protocol_path.read_bytes()
        allowed = json.loads(protocol_bytes)["tasks"][task]["train_state_ids"]
        if len(allowed) != 100 or len(set(allowed)) != 100:
            raise ValueError("Expected the declared100 unique training-state bank")
        episode_bytes = episodes_path.read_bytes()
        episodes = [json.loads(line) for line in episode_bytes.splitlines() if line.strip()]
        if len(episodes) != summary["episodes"]:
            raise ValueError("Final completed-episode counter and episode log differ")
        if any(row["reset"]["experiment_split"] != "train" for row in episodes):
            raise ValueError("Nontraining reset in training episode log")
        counts = Counter(row["reset"]["initial_state_id"] for row in episodes)
        if set(counts) - set(allowed):
            raise ValueError("Training reset outside the frozen training bank")
        record.update(status="complete", final_iterations=100, completed_episode_resets=len(episodes),
                      distinct_states_in_completed_episodes=len(counts), repeated_reset_draws=len(episodes) - len(counts),
                      states_not_in_completed_episode_log=len(allowed) - len(counts),
                      completed_episode_state_ids=sorted(counts),
                      episode_log=str(episodes_path.resolve()), episode_log_sha256=hashlib.sha256(episode_bytes).hexdigest(),
                      final_summary=str(final_path.resolve()), final_summary_sha256=hashlib.sha256(summary_bytes).hexdigest(),
                      protocol_sha256=hashlib.sha256(protocol_bytes).hexdigest(),
                      scope="Completed training episodes only. Resets sample with replacement. A final unfinished episode is not in episodes.jsonl, so its reset is excluded; zero count does not prove a state was never entered.")
        summaries.append(record)
        visits.extend({"task": task, "state_id": state_id, "completed_episode_reset_count": counts[state_id],
                       "episode_log_sha256": record["episode_log_sha256"]} for state_id in allowed)
    return summaries, visits
