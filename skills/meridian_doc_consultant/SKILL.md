---
name: meridian-doc-consultant
description: >-
  Assists users with questions about Meridian Marketing Mix Modeling concepts, parameters, and best practices by finding and consulting the relevant documentation on developers.google.com/meridian. Use when the user asks for explanations or guidance on Meridian topics. Don't use for writing code or running models directly.
---

# Meridian Documentation Consultant

This skill guides the agent in finding and consulting the right page of the
official Meridian documentation to answer user questions token-efficiently.

The documentation is published at `https://developers.google.com/meridian/`. It
is **not** included in the Meridian GitHub repository or the `google-meridian`
Python package, so do not look for documentation files on disk.

## Core Workflow

When the user asks a conceptual or procedural question about Meridian, follow
these steps:

### 1. Identify the Topic

*   Analyze the user's question to identify the core topic (e.g., "knots",
    "priors", "data format").

### 2. Locate Candidate Pages (RAG-style Retrieval)

*   Consult the [Documentation Map](references/documentation_map.md) and use its
    summaries to identify a **batch of potentially relevant pages** (e.g., 3-5
    pages) related to the user's query.
*   **Direct Term Search**: If the user asks about specific technical terms
    (e.g., `eta_m`, `xi_c`, `knots`), search the Documentation Map for these
    terms and then search the text of the candidate pages you fetch in step 3,
    **EVEN IF** a page in the Documentation Map seems to cover the topic. If the
    map has no match, you may use web search restricted to
    `developers.google.com/meridian` to find candidate pages.
*   **Concept Intersection**: If the query involves multiple concepts (e.g.,
    "priors" + "insufficient data"), look for pages that discuss these concepts
    together.
*   **CRITICAL**: Do NOT fetch pages one by one to discover the right one, as
    fetching unrelated pages wastes tokens and latency. Rely on the map's
    summaries to identify candidates.

### 3. Fetch and Read Candidates

*   From the candidate list, select the **Top N (e.g., 3) most promising pages**
    to read.
*   Fetch the **Markdown version** of each page by appending `.md.txt` to its
    URL. For example, the page
    `https://developers.google.com/meridian/docs/pre-modeling/amount-data-needed`
    is available as plain Markdown at
    `https://developers.google.com/meridian/docs/pre-modeling/amount-data-needed.md.txt`.
    Use your URL-fetching tool, or `curl -sL <url>.md.txt` if you only have a
    shell. The regular HTML page is mostly site layout, so prefer the `.md.txt`
    version.
*   For large pages, search the fetched text for specific keywords to locate the
    relevant section instead of reading everything.
*   **Broaden Search Terms**: If a specific keyword search fails (e.g., "maximum
    channels"), try searching for broader related concepts (e.g., "channels") to
    find the relevant section.
*   **Searching Recommendations**: When looking for recommendations, search for
    keywords like "recommend", "best practice", or "should" combined with the
    topic keyword.
*   **Handling Fuzzy Matches on Limits**: If the user asks for a "maximum" and
    the docs say "below X", provide that value.
*   **Prefer Documentation over Code**: Prioritize the documentation pages over
    source code files for conceptual questions.
*   **No Web Access**: If you cannot fetch the pages, tell the user that this
    skill needs access to `developers.google.com/meridian`, give them the page
    URLs from the Documentation Map to read, and do not present unverified
    details as documented guidance.

### 4. Synthesize and Answer

*   **Multi-Document Synthesis**: The answer does not have to depend on a single
    page. Synthesize the answer from multiple candidate pages if they provide
    complementary information.
*   **Grounding Constraint**: Do not answer with specific numbers or
    recommendations from your general knowledge or from search-result snippets
    if they are not present in a page you fetched and read.
*   **Cite the Source**: In your final response, you **MUST** cite the full page
    URL(s) (e.g.,
    `https://developers.google.com/meridian/docs/pre-modeling/amount-data-needed`)
    of all documentation you consulted. Only cite pages you actually fetched and
    read. Do not cite local file paths such as `docs/...md`; they do not exist
    for Meridian users. Ensure you cite the specific page that provided the
    answer to the specific condition.
