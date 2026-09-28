from src.train import run


def test_run_trains_and_evaluates_both_candidate_models():
    summary = run(save=False)

    assert summary["best_model"] in {"logistic_regression", "random_forest"}
    assert set(summary["results"].keys()) == {"logistic_regression", "random_forest"}

    for metrics in summary["results"].values():
        assert 0.0 <= metrics["roc_auc"] <= 1.0
        assert 0.0 <= metrics["recall_attrition"] <= 1.0
        assert len(metrics["confusion_matrix"]) == 2


def test_best_model_is_selected_by_highest_roc_auc():
    summary = run(save=False)
    best_auc = summary["results"][summary["best_model"]]["roc_auc"]
    assert all(m["roc_auc"] <= best_auc for m in summary["results"].values())
