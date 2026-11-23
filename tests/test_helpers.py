"""helpers: the stand-ins for pytest.raises and tmp_path used by the other tests."""

from helpers import raises, uses_tmp_path


def _assertion_message(block):
    try:
        block()
    except AssertionError as err:
        return str(err)
    return None


def test_raises_accepts_the_expected_exception_with_a_matching_message():
    with raises(ValueError, match="bad .* value"):
        raise ValueError("a bad input value")


def test_raises_fails_when_nothing_is_raised():
    def block():
        with raises(ValueError):
            pass

    assert "was not raised" in _assertion_message(block)


def test_raises_fails_when_the_message_does_not_match():
    def block():
        with raises(ValueError, match="expected text"):
            raise ValueError("something else")

    assert "does not match" in _assertion_message(block)


def test_raises_does_not_swallow_a_different_exception_type():
    try:
        with raises(ValueError):
            raise KeyError("k")
    except KeyError:
        return
    raise AssertionError("a KeyError should propagate")


def test_uses_tmp_path_gives_each_call_an_empty_directory_and_removes_it_afterwards():
    seen = []

    @uses_tmp_path
    def body(tmp_path):
        assert tmp_path.is_dir() and not list(tmp_path.iterdir())
        (tmp_path / "x").write_text("x")
        seen.append(tmp_path)

    body()
    body()
    assert seen[0] != seen[1] and not seen[0].exists() and not seen[1].exists()
    assert body.__name__ == "body"
