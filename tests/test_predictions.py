def test_imports_predictions():
    from predictions.tournament_predictor import run_prediction
    assert callable(run_prediction)
