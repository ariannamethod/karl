from utils.tools import sanitize_filename

def test_sanitize_filename_removes_traversal():
    assert sanitize_filename('../secret.txt') == 'secret.txt'
