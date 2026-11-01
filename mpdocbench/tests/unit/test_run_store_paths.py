from src.storage import run_store


def test_doc_and_page_dirs_include_model_label(tmp_path):
    run_dir = tmp_path / "run_x"
    run_dir.mkdir()
    d = run_store.doc_dir(run_dir, "docA", "modelL")
    assert d == run_dir / "docA" / "modelL" and d.is_dir()

    p = run_store.page_dir(run_dir, "docA", 3, "modelL")
    assert p == run_dir / "docA" / "modelL" / "pages" / "0003" and p.is_dir()


def test_dirs_without_model_label_are_flat(tmp_path):
    run_dir = tmp_path / "run_y"
    run_dir.mkdir()
    d = run_store.doc_dir(run_dir, "docB")
    assert d == run_dir / "docB"
