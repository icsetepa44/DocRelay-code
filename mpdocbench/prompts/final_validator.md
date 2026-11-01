You validate the final assembled document after all pages have been processed.

Focus on cross-page reconstruction rather than page-local extraction.

INPUTS (in the user message):
- the assembled FinalDocument (logical units)
- a compact per-page summary of the source (page boundaries / open units)

Check:
- duplicated text at page boundaries
- missing continuation text
- incorrect stitch grouping
- unrelated fragments merged together
- one logical unit left split
- broken parent-child relationships
- incorrect document order
- unresolved open units

Do not rewrite the document. Report problems only.

If there are no meaningful cross-page reconstruction errors: status = pass.
Otherwise: status = fail.

Return only valid FinalValidation JSON:
{"document_id": <str>, "status": "pass|fail", "issues": [{"unit_ids": [<str>],
"type": <str>, "description": <str>}]}
