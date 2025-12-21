def test_imports_data_engine():
    from src.data_engine.load_raw_stats import load_all_player_stats
    from src.data_engine.validate_data import validate_player_data
    from src.data_engine.calculate_talent import estimate_all_talents
    assert callable(load_all_player_stats)
    assert callable(validate_player_data)
    assert callable(estimate_all_talents)
