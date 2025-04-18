# utils_task_2/constants.py

# years and companies you support
ALLOWED_YEARS = ["2018", "2019", "2020"]
ALLOWED_TICKERS = ["aapl", "msft", "goog"]


SECTION_DEFINITIONS = {
    "Business": "A description of the company's business, including its principal products or services, markets served, and competitive conditions.",
    "Risk Factors": "An overview of the material risks that could affect the company's future performance or financial condition.",
    "Unresolved Staff Comments": "Any prior SEC staff comments that remain unresolved at the time of the report.",
    "Properties": "Details about the company's principal physical properties, such as plants, mines, and real estate.",
    "Legal Proceedings": "Information on significant pending legal actions, claims, or governmental investigations.",
    "Mine Safety Disclosures": "Required disclosures concerning mine safety violations and related matters (where applicable).",
    "Market for Common Equity": "Data on the market price of the company's common stock, dividends, and equity holders.",
    "Selected Financial Data": "A five-year summary of key financial data showing results of operations and financial position.",
    "MD&A": "Management's discussion and analysis of financial condition, results of operations, and liquidity.",
    "Market Risk Disclosures": "Quantitative and qualitative disclosures about exposure to market risk (e.g., interest rate, currency).",
    "Financial Statements": "The audited consolidated financial statements and related notes.",
    "Accounting Disagreements": "Disclosures of changes in and disagreements with accountants on financial disclosures.",
    "Controls and Procedures": "Information about the effectiveness of the company's disclosure controls and procedures.",
    "Other Information": "Any other material information required to be reported, not covered elsewhere.",
    "Directors & Governance": "Details about the board of directors, executive officers, and corporate governance practices.",
    "Executive Compensation": "Information on compensation paid to directors and executive officers.",
    "Security Ownership": "Ownership of the company's securities by certain beneficial owners and management.",
    "Related Transactions": "Transactions and relationships between the company and related parties.",
    "Accounting Fees & Services": "Fees paid to the company's independent auditors for audit and non-audit services.",
    "Exhibits & Schedules": "List of exhibits, financial statement schedules, and Form 8-K reports filed."
}


# map human-friendly section names → internal IDs
SECTION_NAME_TO_ID = {
    "Business":                              "section_1",
    "Risk Factors":                          "section_1A",
    "Unresolved Staff Comments":             "section_1B",
    "Properties":                            "section_2",
    "Legal Proceedings":                     "section_3",
    "Mine Safety Disclosures":               "section_4",
    "Market for Common Equity":              "section_5",
    "Selected Financial Data":               "section_6",
    "MD&A":                                  "section_7",
    "Market Risk Disclosures":               "section_7A",
    "Financial Statements":                  "section_8",
    "Accounting Disagreements":              "section_9",
    "Controls and Procedures":               "section_9A",
    "Other Information":                     "section_9B",
    "Directors & Governance":                "section_10",
    "Executive Compensation":                "section_11",
    "Security Ownership":                    "section_12",
    "Related Transactions":                  "section_13",
    "Accounting Fees & Services":            "section_14",
    "Exhibits & Schedules":                  "section_15"
}

# reverse mapping for display
SECTION_ID_TO_NAME = {v: k for k, v in SECTION_NAME_TO_ID.items()}

# SEC-required User-Agent header
SEC_HEADERS = {
    "User-Agent": "AlphaBot/1.2 (api@alpha.example.com)",
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive"
}


TEST_QUERIES = [
    {
        "query": "According to Apple's 2018 Form 10-K, how many weeks did Apple's fiscal year 2017 span?",
        "ground_truth": "53 weeks"
    },
    {
        "query": "What depreciation and amortization expense on property, plant and equipment did Apple report in 2018?",
        "ground_truth": "$9.3 billion"
    },
    {
        "query": "What method did Apple use to determine the cost of securities sold in 2018?",
        "ground_truth": "Specific identification method."
    },
    {
        "query": "In 2018, how does Apple define the length and end-date of its fiscal year?",
        "ground_truth": "A 52- or 53-week period that ends on the last Saturday of September."
    },
    {
        "query": "How were shipping and handling costs classified in Apple's 2018 10-K?",
        "ground_truth": "Classified as cost of sales."
    },
    {
        "query": "In Microsoft's 2019 Form 10-K, how many full-time employees did Microsoft employ?",
        "ground_truth": "144,000 total; 85,000 in the U.S.; 59,000 internationally."
    },
    {
        "query": "In the Business section of Microsoft's 2019 Form 10-K, what mission statement does Microsoft provide?",
        "ground_truth": "To empower every person and every organization on the planet to achieve more."
    },
    {
        "query": "In Microsoft's 2019 Form 10-K Business section, which developer platform acquisition did Microsoft complete on October 25, 2018?",
        "ground_truth": "Acquisition of GitHub, Inc."
    },
    {
        "query": "According to Microsoft's 2019 Form 10-K, what three operating segments does Microsoft use to report its financial performance?",
        "ground_truth": "Productivity and Business Processes; Intelligent Cloud; More Personal Computing."
    },
    {
        "query": "In Alphabet's 2020 Form 10-K, what percentage of the company's total revenues came from the display of ads online?",
        "ground_truth": "Over 80 percents of total revenues."
    },
    {
        "query": "According to Alphabet's 2020 annual report, which global event is cited as having impacted the demand for advertising and advertiser spending cycles?",
        "ground_truth": "COVID-19 and its effects on the global economy."
    },
    {
        "query": "In Alphabet's 2020 Risk Factors section, what ability do advertisers and distribution partners have regarding their contracts with Google?",
        "ground_truth": "They can terminate their contracts at any time."
    },
    {
        "query": "Which kinds of technologies are mentioned in Alphabet's 2020 Risk Factors as potentially impairing or blocking the display of customized ads online?",
        "ground_truth": "Technologies that block ads or make customized ads more difficult to deliver."
    },
    {
        "query": "What competitive threat does Alphabet warn of in its 2020 Risk Factors that could harm its business if it fails to keep innovating?",
        "ground_truth": "Intense competition from large established companies and emerging start-ups."
    }
]
