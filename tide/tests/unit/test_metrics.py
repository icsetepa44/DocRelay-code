import random

from src.eval import evaluate


def _gold(*clauses, tables=()):
    return {"items": [{"type": "clause", "text": c} for c in clauses], "tables": list(tables),
            "full_text": " ".join(clauses), "stats": {}}


def _pred(*clauses, tables=()):
    return {"contents": [{"type": "clause", "text": c, "full_text": c} for c in clauses],
            "tables": list(tables), "adapter_stats": {}}


def test_error_rate_basic():
    assert evaluate.error_rate(["a", "b", "c"], ["a", "b", "c"]) == 0.0
    assert evaluate.error_rate(["a", "b", "c"], ["a", "x", "c"]) == round(1 / 3, 4)
    assert evaluate.error_rate([], []) == 0.0
    assert evaluate.error_rate([], ["x"]) == 1.0
    assert evaluate.error_rate(["a", "b"], []) == 1.0


def test_levenshtein():
    assert evaluate._levenshtein("kitten", "sitting") == 3
    assert evaluate._levenshtein([], [1, 2]) == 2


def test_levenshtein_matches_exact_dp_on_random_bangla_english():
    rng = random.Random(7)
    alphabet = list("abcde ") + list("কখগঘঙচছজ্ািীুেো ")
    for n in (0, 1, 5, 40, 120, 520, 700):
        for _ in range(3):
            a = "".join(rng.choice(alphabet) for _ in range(n))
            b = "".join(rng.choice(alphabet) for _ in range(rng.randint(max(0, n - 30), n + 30)))
            assert evaluate._levenshtein(a, b) == evaluate._levenshtein_dp(a, b)


def test_levenshtein_on_token_lists():
    a = "এই প্রজ্ঞাপন অবিলম্বে কার্যকর হইবে".split() * 120
    b = list(a)
    b[3], b[400] = "ভুল", "শব্দ"
    del b[100]
    assert evaluate._levenshtein(a, b) == evaluate._levenshtein_dp(a, b) == 3


def test_token_bag_similarity_is_order_insensitive():
    a = "the applicant shall provide documents"
    b = "documents provide shall applicant the"
    assert evaluate.token_bag_similarity(a, b) == 1.0
    assert evaluate.token_similarity(a, b) < 1.0
    assert evaluate.token_bag_similarity("", "") == 1.0
    assert evaluate.token_bag_similarity("a b", "") == 0.0
    assert evaluate.token_bag_similarity("a b", "b c") == 0.5


def test_edit_similarity_is_typo_tolerant():
    es = lambda a, b: evaluate.edit_similarity(evaluate.normalize_prose(a), evaluate.normalize_prose(b))
    assert es("the applicant shall provide documents", "the applicant shall provide documents") == 1.0
    assert es("the applicant shall provide the required documents",
              "the applicant shall provide the requred documnts") > 0.9
    assert es("abcdefgh", "12345678") < 0.2
    assert es("", "") == 1.0 and es("abc", "") == 0.0
    a, b = "the quick brown fox", "the quikc brown fox"
    assert es(a, b) == es(b, a)
    assert es("alpha beta gamma delta", "delta gamma beta alpha") < 0.9
    assert evaluate.edit_similarity("a b c", "a c", level="word") == round(1 - 1 / 3, 4)


def test_long_bangla_similarity_is_not_garbage():
    base = "সরকার, কাস্টমস আইন এর ধারা ১৯ এর উপ-ধারা (১) এ প্রদত্ত ক্ষমতাবলে, জনস্বার্থে " * 12
    typo = base.replace("ক্ষমতাবলে", "ক্ষমতা বলে", 1)
    assert len(base) > 500
    assert evaluate.edit_similarity(evaluate.normalize_prose(base), evaluate.normalize_prose(typo)) > 0.99


def test_soft_clause_metrics_present_and_near_perfect():
    m = evaluate.evaluate_pair(
        _gold("The applicant shall submit the required documents within thirty days.",
              "The Commissioner may extend the period for a further fifteen days."),
        _pred("The applicant shall submit the requred documnts within thirty days.",
              "The Commissioner may extend the period for a further fifteen days."))
    assert m["soft_clause_f1"] >= 0.9
    assert 0.0 <= m["soft_clause_precision"] <= 1.0
    assert m["mean_clause_edit_similarity"] > 0.9
    assert "doc_edit_similarity" in m
    assert "clause_f1" in m and "clause_precision" in m


def test_soft_recovers_where_strict_misses():
    m = evaluate.evaluate_pair(_gold("Definitions and interpretation of terms used in these rules."),
                               _pred("Defnitons adn interpretaton of trms usd in thse ruls."))
    assert m["soft_clause_f1"] >= m["clause_f1"]
    assert m["soft_clause_f1"] > 0.5


def test_soft_precision_penalizes_oversegmentation():
    m = evaluate.evaluate_pair(
        _gold("The applicant shall submit the required documents and pay the prescribed fee."),
        _pred("The applicant shall submit the required documents", "and pay the prescribed fee."))
    assert m["gold_clauses"] == 1 and m["pred_clauses"] == 2
    assert m["over_segmentation_factor"] == 2.0
    assert m["soft_clause_precision"] < 1.0


def test_length_gate_uses_normalized_text():
    clause = "The applicant shall submit the required documents within thirty days."
    marked = "**" + " ** ".join(clause.split()) + "** <b></b>" * 20
    assert len(marked) > len(clause) / 0.35
    assert evaluate.evaluate_pair(_gold(clause), _pred(marked))["soft_clause_f1"] == 1.0


_TABLE_KEYS = ("table_teds", "table_teds_struct", "table_grits_con", "table_grits_top")
_TABLE = {"headers": ["h1", "h2"], "rows": [["a", "b"], ["c", "d"]]}


def test_structural_table_keys_none_without_tables():
    m = evaluate.evaluate_pair(_gold("A plain clause with no tables."), _pred("A plain clause with no tables."))
    for k in _TABLE_KEYS:
        assert k in m and m[k] is None


def test_structural_table_keys_perfect_on_identical_table():
    m = evaluate.evaluate_pair(_gold("see the schedule", tables=[_TABLE]), _pred("see the schedule", tables=[_TABLE]))
    for k in _TABLE_KEYS:
        assert m[k] == 1.0
    assert m["gold_tables"] == 1 and "table_cell_recall" in m


def test_structural_table_keys_drop_when_pred_table_missing():
    m = evaluate.evaluate_pair(_gold("see the schedule", tables=[_TABLE]), _pred("see the schedule"))
    for k in _TABLE_KEYS:
        assert m[k] == 0.0
