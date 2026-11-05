import csv
from pathlib import Path

import pytest
from PIL import Image

from rl_vla.diagnostics import causal_figures as plots
from rl_vla.diagnostics import figure_artifacts as original


def source_fixture(root, monkeypatch):
    folder = root / 'causal_analysis/object_0/rldt_000050'
    folder.mkdir(parents=True)
    values = [{'condition': condition, 'states': 50,
               'mean_decoded_controller_l2': float(i),
               'paired_state_bootstrap95_mean': [max(0., i - .1), i + .1]}
              for i, condition in enumerate(plots.CONDITIONS)]
    original.write(folder / 'summary.json', {'complete': True, 'task': 'object_0', 'policy': 'rldt_000050',
                   'open_loop_states': 50, 'open_loop_conditions': 15, 'paired_episodes_per_condition': 10,
                   'closed_loop_conditions': 9, 'open_loop_aggregates': values})
    with (folder / 'open_loop_actions.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['condition', 'state_id', 'l2'])
        writer.writeheader()
        for condition in plots.CONDITIONS:
            writer.writerows({'condition': condition, 'state_id': state, 'l2': .5} for state in range(150, 200))
    for suffix in ('png', 'pdf'):
        Image.new('RGB', (20, 20), 'white').save(folder / f'open_loop_sensitivity.{suffix}')
    source_manifest = original.manifest_path(root, 'causal_analysis', 'object_0', 'rldt_50')
    original.write(source_manifest, {'domain': 'causal_analysis', 'task': 'object_0', 'policy': 'rldt_50'})
    monkeypatch.setattr(original, 'verify_manifest', lambda path, **kwargs: original.read(path))
    return folder, values


def test_no_completion_before_original_validated_manifest(tmp_path):
    folder = tmp_path / 'causal_analysis/object_0/rldt_000050'
    original.write(folder / 'summary.json', {'complete': True})
    assert plots.render_cell(tmp_path, 'object_0', 'rldt_50') is None
    coverage = plots.verify_coverage(tmp_path)
    assert not coverage['complete'] and coverage['passed'] is None
    assert coverage['pending_cells'] == 12 and coverage['failed_cells'] == 0


def test_preserves_exact_statistics_and_binds_all_outputs(tmp_path, monkeypatch):
    folder, values = source_fixture(tmp_path, monkeypatch)
    before = {str(path): original.sha(path) for path in folder.iterdir()}
    manifest = plots.render_cell(tmp_path, 'object_0', 'rldt_50')
    assert manifest['plot_values'] == values
    assert manifest['statistics_recomputed'] is False
    assert {str(path): original.sha(path) for path in folder.iterdir()} == before
    assert len(manifest['artifacts']) == 4
    for artifact in manifest['artifacts']:
        original.artifact(artifact['path'], expected_sha=artifact['sha256'])
    table = next(Path(row['path']) for row in manifest['artifacts'] if row['path'].endswith('.csv'))
    with table.open() as stream:
        rows = list(csv.DictReader(stream))
    for actual, expected in zip(rows, values, strict=True):
        assert actual['condition'] == expected['condition']
        assert float(actual['mean_decoded_controller_l2']) == expected['mean_decoded_controller_l2']
        assert [float(actual['ci95_low']), float(actual['ci95_high'])] == expected['paired_state_bootstrap95_mean']
    assert plots.verify_coverage(tmp_path, full_hash=True)['completed_cells'] == 1
    summary_path = plots.cell_folder(tmp_path, 'object_0', 'rldt_50') / 'summary.json'
    summary = original.read(summary_path)
    summary['artifacts'] = []
    original.write(summary_path, summary)
    coverage = plots.verify_coverage(tmp_path, full_hash=True)
    assert coverage['failed_cells'] == 1 and coverage['passed'] is False


def test_missing_case_and_changed_original_are_rejected(tmp_path, monkeypatch):
    folder, _ = source_fixture(tmp_path, monkeypatch)
    path = folder / 'open_loop_actions.csv'
    body = path.read_text()
    path.write_text(body.rsplit('\n', 2)[0] + '\n')
    with pytest.raises(ValueError, match='coverage'):
        plots.render_cell(tmp_path, 'object_0', 'rldt_50')
    path.write_text(body)
    plots.render_cell(tmp_path, 'object_0', 'rldt_50')
    path.write_text(body + '\n')
    with pytest.raises(ValueError, match='changed'):
        plots.verify_cell(tmp_path, 'object_0', 'rldt_50', full_hash=True)
