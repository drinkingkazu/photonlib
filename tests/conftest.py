
pytest_plugins = "tests.fixtures"

# import pytest
# def pytest_collection_modifyitems(config, items):
#     for item in items:
#         if "apps" in item.fspath.strpath:
#             item.add_marker(pytest.mark.apps)