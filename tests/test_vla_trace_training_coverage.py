import json

import pytest

from rl_vla.diagnostics.training_coverage import final_training_reset_coverage


def test_final_training_coverage_counts_repeats_not_prescribed_bank(tmp_path):
    directory = tmp_path / "methods/rldt/object_0/training"
    directory.mkdir(parents=True)
    protocol = tmp_path / "tasks/object_0/protocol.json"
    protocol.parent.mkdir(parents=True)
    protocol.write_text(json.dumps({"tasks": {"object_0": {"train_state_ids": list(range(100))}}}))
    episodes = [{"reset": {"initial_state_id": state, "experiment_split": "train"}} for state in (5, 5, 12)]
    (directory / "episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in episodes))
    # A partially completed run cannot be presented as final coverage.
    assert final_training_reset_coverage(tmp_path, ("object_0",))[0][0]["status"] == "pending_final_training_summary"
    (directory / "summary.json").write_text(json.dumps({"iterations": 100, "episodes": 3}))
    summaries, visits = final_training_reset_coverage(tmp_path, ("object_0",))
    assert summaries[0]["completed_episode_resets"] == 3
    assert summaries[0]["distinct_states_in_completed_episodes"] == 2
    assert summaries[0]["repeated_reset_draws"] == 1
    assert len(visits) == 100 and visits[5]["completed_episode_reset_count"] == 2
    (directory / "summary.json").write_text(json.dumps({"iterations": 100, "episodes": 4}))
    with pytest.raises(ValueError, match="counter"):
        final_training_reset_coverage(tmp_path, ("object_0",))
