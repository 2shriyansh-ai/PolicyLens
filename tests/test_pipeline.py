from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from policylens.answering import answer_question
from policylens.chunking import fixed_size_chunks, structure_aware_chunks
from policylens.evaluation import correct_rank
from policylens.dataset import validate_cases
from policylens.documents import extract_pages
from policylens.experiments import choose_threshold, threshold_sweep
from policylens.failures import add_failure_categories
from policylens.grading import lexical_reference_grade
from policylens.models import Chunk, EvaluationCase, Page
from policylens.retrieval import HashingEmbedder, PolicyIndex


class ChunkingTests(unittest.TestCase):
    def setUp(self):
        self.pages = [
            Page(
                document="leave.txt",
                number=1,
                text=(
                    "ANNUAL LEAVE\n\nEmployees receive twenty days.\n\n"
                    "CARRY FORWARD\n\nFive unused days may be carried forward."
                ),
            )
        ]

    def test_fixed_chunks_preserve_metadata(self):
        chunks = fixed_size_chunks(self.pages, chunk_words=6, overlap_words=2)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0].document, "leave.txt")
        self.assertEqual(chunks[0].pages, (1,))

    def test_structure_chunks_preserve_sections(self):
        chunks = structure_aware_chunks(self.pages, max_words=30)
        self.assertTrue(any("Five unused days" in chunk.text for chunk in chunks))


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        chunks = [
            Chunk("leave", "leave.txt", (8,), "Employees may carry forward five unused leave days."),
            Chunk("travel", "travel.txt", (3,), "Hotels are reimbursed up to INR 5000."),
            Chunk("medical", "leave.txt", (11,), "Medical leave requires a certificate."),
        ]
        self.index = PolicyIndex(chunks, HashingEmbedder(256))

    def test_keyword_search_ranks_matching_passage_first(self):
        results = self.index.keyword_search("five unused leave days", top_k=3)
        self.assertEqual(results[0].chunk.id, "leave")

    def test_answer_contains_source_evidence(self):
        answer = answer_question(
            self.index,
            "How many unused leave days can employees carry forward?",
            method="keyword",
            threshold=0,
        )
        self.assertEqual(answer.status, "answered")
        self.assertEqual(answer.evidence[0].chunk.id, "leave")

    def test_correct_rank_uses_labelled_page(self):
        case = EvaluationCase(
            id="case",
            split="test",
            question="question",
            expected_answer="five",
            relevant_documents=("leave.txt",),
            relevant_pages=(8,),
            question_type="direct",
            answerable=True,
        )
        results = self.index.keyword_search("medical certificate", top_k=3)
        self.assertEqual(correct_rank(case, results), 2)

    def test_unanswerable_question_can_be_refused(self):
        answer = answer_question(
            self.index,
            "Does the company provide a gym membership?",
            method="hybrid_rerank",
            threshold=0.9,
        )
        self.assertEqual(answer.status, "refused")


class EvaluationTests(unittest.TestCase):
    def test_dataset_validator_rejects_missing_evidence(self):
        case = EvaluationCase(
            id="bad",
            split="development",
            question="A question?",
            expected_answer="An answer",
            relevant_documents=(),
            relevant_pages=(),
            question_type="direct",
            answerable=True,
        )
        issues = validate_cases([case])
        self.assertTrue(any(issue.level == "error" for issue in issues))

    def test_threshold_choice_penalizes_false_refusals(self):
        curve = pd.DataFrame(
            [
                {"threshold": 0.1, "correct_refusal_rate": 0.5, "false_refusal_rate": 0.0},
                {"threshold": 0.5, "correct_refusal_rate": 1.0, "false_refusal_rate": 0.5},
            ]
        )
        self.assertEqual(choose_threshold(curve), 0.1)

    def test_failure_classifier_marks_false_refusal(self):
        frame = pd.DataFrame([{"false_refusal": True, "answerable": True, "refused": True}])
        enriched = add_failure_categories(frame)
        self.assertEqual(enriched.iloc[0]["failure_category"], "false_refusal")

    def test_correct_refusal_is_not_a_failure(self):
        frame = pd.DataFrame([{"false_refusal": False, "answerable": False, "refused": True}])
        enriched = add_failure_categories(frame)
        self.assertEqual(enriched.iloc[0]["failure_category"], "none")

    def test_lexical_grader_exposes_proxy_method(self):
        grade = lexical_reference_grade("five unused days", "Employees may carry five unused days")
        self.assertEqual(grade.score, 2)
        self.assertEqual(grade.method, "lexical-proxy")


class PdfTests(unittest.TestCase):
    def test_pdf_text_and_page_number_are_extracted(self):
        import pymupdf

        temporary = ROOT / "tests" / "_generated_policy.pdf"
        document = pymupdf.open()
        page = document.new_page()
        page.insert_text((72, 72), "Employees receive twenty leave days.")
        document.save(temporary)
        document.close()
        try:
            pages = extract_pages(temporary)
            self.assertEqual(pages[0].number, 1)
            self.assertIn("twenty leave days", pages[0].text)
        finally:
            temporary.unlink(missing_ok=True)

    def test_controlled_benchmark_labels_reference_existing_pages(self):
        from policylens.evaluation import load_cases

        cases = load_cases(ROOT / "data" / "benchmark_cases.csv")
        pdf_directory = ROOT / "data" / "benchmark_policies"
        pages_by_document = {
            path.name: {page.number for page in extract_pages(path)}
            for path in pdf_directory.glob("*.pdf")
        }
        self.assertEqual(len(cases), 100)
        self.assertEqual(sum(case.split == "development" for case in cases), 70)
        self.assertEqual(sum(case.split == "test" for case in cases), 30)
        for case in cases:
            if not case.answerable:
                continue
            self.assertIn(case.relevant_documents[0], pages_by_document)
            self.assertTrue(set(case.relevant_pages) <= pages_by_document[case.relevant_documents[0]])


if __name__ == "__main__":
    unittest.main()
