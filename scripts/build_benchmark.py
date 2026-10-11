"""Build PolicyLens's original controlled policy corpus and labelled benchmark.

The corpus is intentionally synthetic and redistributable. It tests the full
pipeline without presenting third-party documents or generated labels as a
real-world accuracy claim.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pymupdf


ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = ROOT / "data" / "benchmark_policies"
CASE_PATH = ROOT / "data" / "benchmark_cases.csv"


DOCUMENTS = {
    "leave_policy.pdf": [
        ("Annual leave", [
            "Employees receive 20 paid annual leave days per calendar year.",
            "Leave accrues at 1.67 days for every completed month of service.",
        ]),
        ("Carry-forward", [
            "A maximum of 5 unused annual leave days may be carried into the next calendar year.",
            "Carried leave expires after 31 March and cannot be encashed.",
        ]),
        ("Probation and medical leave", [
            "New employees become eligible for paid annual leave after completing 6 months of probation.",
            "Medical leave longer than 2 consecutive working days requires a medical certificate.",
        ]),
        ("Family leave", [
            "Primary caregivers receive 26 weeks of paid parental leave.",
            "Secondary caregivers receive 4 weeks of paid parental leave.",
            "Employees receive 3 paid bereavement days for an immediate family member.",
        ]),
    ],
    "remote_work_policy.pdf": [
        ("Eligibility", [
            "Employees may apply for regular remote work after 3 completed months of employment.",
            "Eligible employees may work remotely up to 3 days in a standard workweek.",
        ]),
        ("Availability", [
            "Remote employees must be available during core hours from 10:00 to 16:00 local time.",
            "Messages from colleagues must be acknowledged within 2 working hours during core hours.",
        ]),
        ("Equipment and connectivity", [
            "The company provides approved home-office equipment worth up to INR 60,000.",
            "Employees may claim up to INR 1,500 per month for home internet with a valid invoice.",
        ]),
        ("Working outside India", [
            "Remote work outside India is limited to 20 calendar days in a year.",
            "International remote work requires manager, HR, and information-security approval before travel.",
        ]),
    ],
    "travel_policy.pdf": [
        ("Approval and booking", [
            "All business travel requires manager approval before booking.",
            "Domestic flights should be booked at least 14 days before departure when practical.",
        ]),
        ("Domestic reimbursement limits", [
            "EXPENSE | MAXIMUM REIMBURSEMENT",
            "Hotel | INR 5,000 per night",
            "Meals | INR 1,000 per day",
            "Local transport | Actual cost with valid receipts",
        ]),
        ("Expense claims", [
            "Travel expense claims must be submitted within 30 days after returning from the trip.",
            "Receipts are required for every individual expense above INR 500.",
        ]),
        ("International travel", [
            "Foreign-currency expenses are converted using the card statement rate or the company's monthly rate.",
            "Business-class travel is allowed only when the scheduled flight time exceeds 8 hours and a vice president approves it.",
        ]),
    ],
    "information_security_policy.pdf": [
        ("Account security", [
            "Multi-factor authentication is mandatory for email, VPN, and production systems.",
            "Passwords must contain at least 14 characters and must not be shared.",
        ]),
        ("Incident reporting", [
            "Suspected security incidents must be reported to the security team within 1 hour of discovery.",
            "Employees must not investigate suspected malware on their own device.",
        ]),
        ("Data classification", [
            "Company information is classified as Public, Internal, or Confidential.",
            "Confidential information may be shared externally only with authorization and an approved secure channel.",
        ]),
        ("Storage and retention", [
            "Business records covered by the retention schedule are kept for 7 years.",
            "Confidential company data must not be stored on personal removable media.",
        ]),
    ],
    "benefits_policy.pdf": [
        ("Health insurance", [
            "The group medical policy provides coverage of INR 500,000 per family each year.",
            "Coverage includes the employee, one spouse or partner, and up to 2 dependent children.",
        ]),
        ("Wellness", [
            "Employees may claim up to INR 12,000 per calendar year for eligible wellness expenses.",
            "Gym memberships and fitness classes qualify when supported by a receipt.",
        ]),
        ("Learning", [
            "The annual professional-learning allowance is INR 30,000 per employee.",
            "Courses require manager approval before enrolment to qualify for reimbursement.",
        ]),
        ("Long-term and support benefits", [
            "The company contributes 12 percent of eligible basic salary to the retirement plan.",
            "The employee assistance programme provides 6 confidential counselling sessions per year.",
        ]),
    ],
}


# document, page, expected, direct question, paraphrase, category
FACTS = [
    ("leave_policy.pdf", 1, "20 paid annual leave days", "How many paid annual leave days do employees receive?", "What is the yearly vacation allowance?", "direct"),
    ("leave_policy.pdf", 1, "1.67 days per completed month", "At what monthly rate does annual leave accrue?", "How much vacation is earned after each full month?", "direct"),
    ("leave_policy.pdf", 2, "5 days", "How many unused leave days may be carried forward?", "What is the maximum holiday balance transferable to next year?", "direct"),
    ("leave_policy.pdf", 2, "After 31 March", "When does carried leave expire?", "By what date must last year's transferred vacation be used?", "direct"),
    ("leave_policy.pdf", 3, "After 6 months of probation", "When do new employees become eligible for paid annual leave?", "Can a new hire take paid vacation before finishing six months?", "direct"),
    ("leave_policy.pdf", 3, "More than 2 consecutive working days", "When is a medical certificate required?", "After how many continuous sick days must proof from a doctor be supplied?", "direct"),
    ("leave_policy.pdf", 4, "26 weeks", "How much paid parental leave does a primary caregiver receive?", "What is the parental-leave duration for the main caregiver?", "direct"),
    ("leave_policy.pdf", 4, "4 weeks", "How much paid parental leave does a secondary caregiver receive?", "How long can the supporting caregiver take as parental leave?", "direct"),
    ("remote_work_policy.pdf", 1, "After 3 completed months", "When may an employee apply for regular remote work?", "How long must a new hire wait before requesting work from home?", "direct"),
    ("remote_work_policy.pdf", 1, "Up to 3 days per standard workweek", "How many days per week may eligible employees work remotely?", "What is the weekly work-from-home limit?", "direct"),
    ("remote_work_policy.pdf", 2, "10:00 to 16:00 local time", "What are the remote-work core hours?", "During which hours must home-based employees be available?", "direct"),
    ("remote_work_policy.pdf", 2, "Within 2 working hours", "How quickly must remote employees acknowledge messages?", "What is the response-time expectation during core hours?", "direct"),
    ("remote_work_policy.pdf", 3, "INR 60,000", "What is the maximum home-office equipment allowance?", "How much equipment will the company provide for a home workspace?", "table"),
    ("remote_work_policy.pdf", 3, "INR 1,500 per month", "What is the monthly home-internet reimbursement limit?", "How much can an employee claim each month for broadband?", "table"),
    ("remote_work_policy.pdf", 4, "20 calendar days per year", "How long may an employee work remotely outside India?", "What is the annual overseas work-from-anywhere limit?", "direct"),
    ("remote_work_policy.pdf", 4, "Manager, HR, and information-security approval", "Which approvals are required for international remote work?", "Who must approve working remotely from another country?", "direct"),
    ("travel_policy.pdf", 1, "Manager approval before booking", "What approval is required before business travel is booked?", "Can an employee purchase a business trip before asking a manager?", "direct"),
    ("travel_policy.pdf", 1, "At least 14 days before departure", "When should domestic flights be booked?", "How early should staff normally reserve an internal flight?", "direct"),
    ("travel_policy.pdf", 2, "INR 5,000 per night", "What is the domestic hotel reimbursement limit?", "How much may an employee claim for one hotel night?", "table"),
    ("travel_policy.pdf", 2, "INR 1,000 per day", "What is the daily meal reimbursement limit?", "What is the maximum food allowance on a business trip?", "table"),
    ("travel_policy.pdf", 2, "Actual cost with valid receipts", "How is local transport reimbursed?", "Can a traveller recover the full taxi cost and what proof is required?", "table"),
    ("travel_policy.pdf", 3, "Within 30 days after returning", "When must a travel expense claim be submitted?", "How long after a trip does an employee have to file expenses?", "direct"),
    ("travel_policy.pdf", 3, "For individual expenses above INR 500", "When are receipts mandatory for travel expenses?", "At what amount does an expense need documentary proof?", "table"),
    ("travel_policy.pdf", 4, "Flights exceeding 8 hours with vice-president approval", "When is business-class travel allowed?", "What journey length and approval permit a premium cabin?", "direct"),
    ("information_security_policy.pdf", 1, "Email, VPN, and production systems", "Where is multi-factor authentication mandatory?", "Which company services require a second login factor?", "direct"),
    ("information_security_policy.pdf", 1, "At least 14 characters", "What is the minimum password length?", "How long must a company password be?", "table"),
    ("information_security_policy.pdf", 2, "Within 1 hour of discovery", "How quickly must a suspected security incident be reported?", "What is the reporting deadline after noticing a possible breach?", "direct"),
    ("information_security_policy.pdf", 2, "Do not investigate it independently", "Should employees investigate suspected malware themselves?", "What should a worker avoid doing after finding possible malware?", "direct"),
    ("information_security_policy.pdf", 3, "Public, Internal, or Confidential", "What are the three data-classification levels?", "How can company information be categorised?", "direct"),
    ("information_security_policy.pdf", 3, "Authorization and an approved secure channel", "What is required to share confidential information externally?", "What controls apply before sensitive company data is sent outside?", "direct"),
    ("information_security_policy.pdf", 4, "7 years", "How long are covered business records retained?", "What is the retention period for scheduled company records?", "table"),
    ("information_security_policy.pdf", 4, "It is prohibited", "May confidential company data be stored on personal removable media?", "Can sensitive work files be copied to a personal USB drive?", "direct"),
    ("benefits_policy.pdf", 1, "INR 500,000 per family per year", "What is the annual group medical coverage?", "How much health-insurance protection does each family receive?", "table"),
    ("benefits_policy.pdf", 1, "Employee, one spouse or partner, and up to 2 dependent children", "Who is covered by the group medical policy?", "Which family members can be included in company health insurance?", "direct"),
    ("benefits_policy.pdf", 2, "INR 12,000 per calendar year", "What is the annual wellness reimbursement limit?", "How much can an employee claim yearly for wellbeing expenses?", "table"),
    ("benefits_policy.pdf", 2, "Yes, with a receipt", "Do gym memberships qualify as wellness expenses?", "Can the wellness allowance pay for a fitness-centre membership?", "direct"),
    ("benefits_policy.pdf", 3, "INR 30,000 per year", "What is the professional-learning allowance?", "How much annual funding is available for employee courses?", "table"),
    ("benefits_policy.pdf", 3, "Manager approval before enrolment", "What approval is needed for course reimbursement?", "When must a manager approve training for it to be paid?", "direct"),
    ("benefits_policy.pdf", 4, "12 percent of eligible basic salary", "How much does the company contribute to retirement?", "What percentage of basic pay goes into the employer retirement contribution?", "table"),
    ("benefits_policy.pdf", 4, "6 confidential sessions per year", "How many counselling sessions does the assistance programme provide?", "What confidential mental-health support is available annually?", "table"),
]


MULTI_PAGE = [
    ("leave_policy.pdf", "1|2", "20 days annually, with at most 5 unused days carried forward", "Summarize the annual leave allowance and carry-forward limit."),
    ("leave_policy.pdf", "2|3", "Carried leave expires after 31 March; new hires qualify after 6 months", "What deadlines affect carried leave and a new hire's first paid leave?"),
    ("leave_policy.pdf", "3|4", "A certificate is required after 2 sick days; primary parental leave is 26 weeks", "Compare the medical-proof rule with primary-caregiver leave."),
    ("remote_work_policy.pdf", "1|2", "Up to 3 remote days, with availability from 10:00 to 16:00", "What weekly remote schedule and daily availability rule apply?"),
    ("remote_work_policy.pdf", "3|4", "INR 1,500 monthly internet support; overseas remote work is limited to 20 days", "What support and annual limit apply to an employee working remotely abroad?"),
    ("travel_policy.pdf", "1|3", "Manager approval before booking; claims within 30 days after returning", "State the approval and claim deadlines for business travel."),
    ("travel_policy.pdf", "2|3", "Meals are capped at INR 1,000 daily; receipts are required above INR 500", "What meal limit and receipt threshold apply on a trip?"),
    ("information_security_policy.pdf", "1|2", "MFA is mandatory; suspected incidents must be reported within 1 hour", "What account control and incident deadline protect company systems?"),
    ("information_security_policy.pdf", "3|4", "Confidential data needs authorization and secure transfer and cannot be stored on personal removable media", "How may confidential information be shared and where must it not be stored?"),
    ("benefits_policy.pdf", "1|2", "INR 500,000 family medical cover and INR 12,000 annual wellness reimbursement", "Compare the annual medical and wellness benefits."),
]


UNANSWERABLE = [
    "Does the company provide a free daily lunch?",
    "What is the employee stock-option vesting schedule?",
    "Is there a company car programme?",
    "How many paid volunteering days are provided?",
    "Does the company reimburse pet insurance?",
    "What is the relocation allowance for a new hire?",
    "Are employees given free mobile phones?",
    "What is the dress code on Fridays?",
    "Does the company offer childcare at the office?",
    "How much is the annual performance bonus?",
]


def create_pdf(name: str, pages: list[tuple[str, list[str]]]) -> None:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    document = pymupdf.open()
    for page_number, (heading, lines) in enumerate(pages, 1):
        page = document.new_page(width=595, height=842)
        page.insert_text((54, 54), "POLICYLENS CONTROLLED BENCHMARK", fontsize=8, color=(0.35, 0.4, 0.5))
        page.insert_text((54, 92), heading.upper(), fontsize=18, color=(0.05, 0.2, 0.45))
        y = 132
        for line in lines:
            page.insert_textbox((54, y, 540, y + 72), line, fontsize=11, lineheight=1.35)
            y += 74
        page.insert_text((500, 812), f"Page {page_number}", fontsize=9)
    document.set_metadata({"title": name.removesuffix(".pdf").replace("_", " ").title(), "author": "PolicyLens project"})
    document.save(PDF_DIR / name)
    document.close()


def split_for(index: int) -> str:
    return "test" if index % 10 in {7, 8, 9} else "development"


def build_cases() -> list[dict[str, str]]:
    cases: list[dict[str, str]] = []
    for document, page, expected, direct, paraphrase, category in FACTS:
        for question, question_type in ((direct, category), (paraphrase, "table" if category == "table" else "paraphrased")):
            index = len(cases)
            cases.append({
                "id": f"case_{index + 1:03d}",
                "split": split_for(index),
                "question": question,
                "expected_answer": expected,
                "relevant_documents": document,
                "relevant_pages": str(page),
                "question_type": question_type,
                "answerable": "true",
            })
    for document, pages, expected, question in MULTI_PAGE:
        index = len(cases)
        cases.append({
            "id": f"case_{index + 1:03d}", "split": split_for(index), "question": question,
            "expected_answer": expected, "relevant_documents": document, "relevant_pages": pages,
            "question_type": "multi_page", "answerable": "true",
        })
    for question in UNANSWERABLE:
        index = len(cases)
        cases.append({
            "id": f"case_{index + 1:03d}", "split": split_for(index), "question": question,
            "expected_answer": "Not found in the supplied policies", "relevant_documents": "",
            "relevant_pages": "", "question_type": "unanswerable", "answerable": "false",
        })
    assert len(cases) == 100
    return cases


def main() -> None:
    for name, pages in DOCUMENTS.items():
        create_pdf(name, pages)
    cases = build_cases()
    with CASE_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cases[0]))
        writer.writeheader()
        writer.writerows(cases)
    print(f"Created {len(DOCUMENTS)} PDFs in {PDF_DIR}")
    print(f"Created {len(cases)} labelled cases at {CASE_PATH}")


if __name__ == "__main__":
    main()

