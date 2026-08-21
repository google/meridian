---
name: meridian-doc-consultant
description: >-
  Assists users with questions about Meridian Marketing Mix Modeling concepts, parameters, and best practices by finding and consulting the relevant documentation. Use when the user asks for explanations or guidance on Meridian topics. Don't use for writing code or running models directly.
---

# Meridian Documentation Consultant

This skill guides the agent in finding and consulting the right Meridian
documentation file to answer user questions token-efficiently.

## Core Workflow

When the user asks a conceptual or procedural question about Meridian, follow
these steps:

### 1. Identify the Topic

*   Analyze the user's question to identify the core topic (e.g., "knots",
    "priors", "data format").

### 2. Locate Candidate Documents (RAG-style Retrieval)

*   Consult the [Documentation Map](references/documentation_map.md) and use its
    summaries to identify a **batch of potentially relevant documents** (e.g.,
    3-5 files) related to the user's query.
*   **Direct Term Search**: If the user asks about specific technical terms
    (e.g., `eta_m`, `xi_c`, `knots`), you **MUST** use `grep_search` or text
    search to search for these terms directly across all documentation files to
    find candidates, **EVEN IF** a file in the Documentation Map seems to cover
    the topic.
*   **Concept Intersection**: If the query involves multiple concepts (e.g.,
    "priors" + "insufficient data"), look for files that discuss these concepts
    together.
*   **CRITICAL**: Do NOT read documentation files one by one to discover the
    right one. Rely on the map's summaries and keyword search to identify
    candidates.

### 3. Select and Read Candidates

*   From the candidate list, select the **Top N (e.g., 3) most promising files**
    to read.
*   For each selected file, do NOT read the whole file if it is large.
*   Use `grep_search` or text search to find specific keywords within the file
    to locate the relevant section.
*   **Broaden Search Terms**: If a specific keyword search fails (e.g., "maximum
    channels"), try searching for broader related concepts (e.g., "channels") to
    find the relevant section.
*   **Searching Recommendations**: When looking for recommendations, search for
    keywords like "recommend", "best practice", or "should" combined with the
    topic keyword.
*   **Handling Fuzzy Matches on Limits**: If the user asks for a "maximum" and
    the docs say "below X", provide that value.
*   **Prefer Documentation over Code**: Prioritize reading documentation files
    over source code files for conceptual questions.
*   Use `view_file` with `StartLine` and `EndLine` to read only a window of
    lines around the match (e.g., 50 lines before and after).
*   **Expand the Window if Needed**: If the viewed content seems to continue or
    if you need more context, use `view_file` again to read subsequent or
    preceding lines.

### 4. Synthesize and Answer

*   **Multi-Document Synthesis**: The answer does not have to depend on a single
    document. Synthesize the answer from multiple candidate documents if they
    provide complementary information.
*   **Grounding Constraint**: Do not answer with specific numbers or
    recommendations from your general knowledge if they are not present in the
    consulted documentation.
*   **Cite the Source**: In your final response, you **MUST** explicitly cite
    the full file path(s) of all documentation you consulted. Ensure you cite
    the specific file that provided the answer to the specific condition.
