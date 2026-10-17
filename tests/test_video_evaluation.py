from tools.evaluate_video import score_events


def test_video_metrics_count_false_repeats_and_misses():
    expected = [
        {"label": "A", "start": 0, "end": 1},
        {"label": "B", "start": 2, "end": 3},
    ]
    pred = [
        {"label": "A", "time": 0.5},
        {"label": "A", "time": 0.7},
        {"label": "J", "time": 4},
    ]
    r = score_events(expected, pred)
    assert r["matched"] == 1 and r["missed"] == 1 and r["false_or_repeated"] == 2
    assert r["latencies"][0]["latency_from_start_s"] == 0.5
