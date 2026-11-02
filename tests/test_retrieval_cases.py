from examples.retrieval_cases import retrieval_cases


class ByteTokenizer:
    def encode(self,text,**kwargs):return list(text.encode())


def test_fixed_retrieval_matrix_contains_needle_and_exact_lengths():
    cases=retrieval_cases(ByteTokenizer())
    assert len(cases)==6 and cases==retrieval_cases(ByteTokenizer())
    for c in cases:
        text=bytes(c['tokens']).decode()
        assert len(c['tokens'])==c['length']
        assert f"Record {c['key']} has access code {c['answer']}" in text
        assert f"access code for {c['key']}" in text
        assert len(c['token_sha256'])==64
