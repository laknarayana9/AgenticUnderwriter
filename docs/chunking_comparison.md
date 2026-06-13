# Chunking Strategy Comparison

Corpus retrieval quality for header-based vs fixed-size chunking, measured over 8 probe queries at k=5.

| Strategy | Retrieval mode | Chunks | Mean chars | Hit@k | MRR |
| --- | --- | ---: | ---: | ---: | ---: |
| header | lexical | 61 | 199 | 0.625 | 0.531 |
| header | semantic | 61 | 199 | 0.625 | 0.625 |
| header | hybrid | 61 | 199 | 0.625 | 0.562 |
| fixed | lexical | 25 | 703 | 1.000 | 0.754 |
| fixed | semantic | 25 | 703 | 0.750 | 0.656 |
| fixed | hybrid | 25 | 703 | 0.750 | 0.667 |

## Takeaways

- **header**: best hit@k=0.625 (mrr=0.625) in semantic mode; 61 chunks @ ~199 chars each.
- **fixed**: best hit@k=1.000 (mrr=0.754) in lexical mode; 25 chunks @ ~703 chars each.
- **header** reranking impact (hybrid vs lexical MRR): +0.031.
- **fixed** reranking impact (hybrid vs lexical MRR): -0.088.
