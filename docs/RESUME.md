# Resume and interview wording

## Resume-ready project entry

**PolicyLens — Evaluated RAG system for policy documents**  
Python, Streamlit, sentence-transformers, BM25, PyMuPDF, Pandas

- Built a document question-answering application that retrieves relevant policy passages, returns one evidence-backed answer with page citations, and refuses when supporting evidence is weak.
- Created a reproducible 100-question controlled benchmark with development and held-out test splits; the selected semantic retriever achieved 27/27 Hit@3 and 0.938 MRR on the 27 answerable held-out cases, while failure analysis identified false refusals as the main remaining limitation.
- Compared fixed and structure-aware chunking plus semantic, hybrid, and reranked retrieval while tracking retrieval quality, refusal behaviour, faithfulness, latency, cost, and bootstrap confidence intervals.

Keep the words **controlled benchmark** in the resume or README. The results are real for the included test corpus, but they are not a claim that PolicyLens will perform equally well on every company's documents.

## 30-second interview explanation

“PolicyLens lets an employee upload policy PDFs and ask a normal-language question. It splits the documents into passages, converts those passages into embeddings, and retrieves the passages most related to the question. If the evidence is strong enough, it returns one answer with the document name and page number; otherwise, it says that it could not find enough evidence. I also built a separate evaluation lab so I could compare retrieval methods on held-out questions instead of judging the system from a few good demos.”

## If asked what makes it different from NotebookLM

“NotebookLM is a broad, polished research assistant. PolicyLens is a focused engineering project for auditable policy Q&A. Its value is not that it has more features; it is that the retrieval, refusal threshold, citations, held-out evaluation, latency, and failure cases are visible and measurable. I built it to understand and demonstrate how a reliable document-Q&A system is tested.”

## Honest limitation to mention

The strongest configuration retrieved the correct passage in its first three results for every answerable held-out question, but it falsely refused 7 of 27 answerable questions at the selected threshold. That means retrieval was strong while abstention calibration still needs improvement. A good next step is a larger, independently reviewed dataset with more diverse document layouts.
