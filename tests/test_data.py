from src.data import DROP_COLUMNS, FEATURES, TARGET, load_clean, load_raw, split_features_target


def test_load_raw_has_expected_shape():
    df = load_raw()
    assert df.shape == (1470, 35)
    assert "Attrition" in df.columns


def test_load_clean_drops_uninformative_columns():
    df = load_clean()
    for col in DROP_COLUMNS:
        assert col not in df.columns


def test_load_clean_encodes_target_as_binary():
    df = load_clean()
    assert set(df[TARGET].unique()) == {0, 1}


def test_split_features_target_matches_feature_list():
    df = load_clean()
    X, y = split_features_target(df)
    assert list(X.columns) == FEATURES
    assert y.name == TARGET
    assert len(X) == len(y)
