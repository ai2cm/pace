import doctest

import pace.driver.registry


def test_registry_doc_examples():
    result = doctest.testmod(pace.driver.registry)
    assert result.attempted > 0, "No doctests found"
    assert result.failed == 0, "doctests failed"
