def test_imports_course_engine():
    from src.course_engine.calculate_fits import calculate_all_course_fits
    assert callable(calculate_all_course_fits)
