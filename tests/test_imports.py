"""Verify that every public module can be imported without errors."""


def test_import_src():
    import src  # noqa: F401


def test_import_data_loader():
    import src.data_loader  # noqa: F401


def test_import_sentiment():
    import src.sentiment  # noqa: F401


def test_import_indicators():
    import src.indicators  # noqa: F401


def test_import_visualization():
    import src.visualization  # noqa: F401


def test_import_utils():
    import src.utils  # noqa: F401


def test_import_eda():
    import src.eda  # noqa: F401


def test_import_correlation():
    import src.correlation  # noqa: F401
