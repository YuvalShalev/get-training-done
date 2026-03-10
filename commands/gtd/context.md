---
description: Provide domain context for a dataset before EDA or training
argument-hint: "path/to/data.csv [free-text context description]"
---

# GTD: Dataset Context

You are collecting domain context from the user to improve EDA analysis, research queries, and training decisions.

## CRITICAL RULES
1. This command is lightweight — no MCP tools needed
2. Use AskUserQuestion for all user interaction
3. Write the context artifact using the Write tool

## Argument Parsing

The user's arguments are: `$ARGUMENTS`

Parse from the argument string:
- `DATA_PATH` (required) — path to the CSV file (first argument)
- Remaining text after the path = free-text domain context (optional)

If no free-text context is provided, use AskUserQuestion to ask:
"Describe the domain and problem for this dataset. For example: 'vehicle health prediction using sensor readings' or 'medical diagnosis from patient records'."

---

## Workflow

### Step 1: Collect Context

Read the user's free-text context (from arguments or AskUserQuestion).

### Step 2: Adaptive Follow-Up Questions

Based on what the user said, ask **1-2 targeted follow-up questions** via AskUserQuestion. Pick questions that are relevant to the domain they described:

- If user mentions **sensors/IoT**: "What is the sampling frequency, and are readings from multiple sensors at the same timestamp?"
- If user mentions **medical/clinical**: "Is the data at the patient level, or are there multiple records per patient (longitudinal)?"
- If user mentions **time series/temporal**: "What is the prediction horizon (e.g., next hour, next day, next month)?"
- If user mentions **financial/trading**: "Is this point-in-time data, or do you need to avoid look-ahead bias?"
- If user mentions **NLP/text**: "Are the text fields pre-processed, or do they need tokenization/embedding?"
- If user mentions **images/vision**: "Are the image features pre-extracted, or is this raw pixel data?"
- If user mentions **recommendation/ranking**: "Is there implicit feedback (clicks) or explicit feedback (ratings)?"
- If user mentions **fraud/anomaly**: "What is the approximate fraud/anomaly rate in the data?"
- If user mentions **manufacturing/quality**: "Are measurements from a single production line or multiple lines/machines?"
- If user mentions **customer/churn**: "What is the observation window and prediction window for churn definition?"

If the domain doesn't match any of the above, ask a general follow-up:
"Any specific challenges or constraints I should know about? (e.g., class imbalance, missing data patterns, regulatory requirements, real-time inference needs)"

Combine the original context + follow-up answers into a single concise domain description.

### Step 3: Extract Keywords

From the combined context, extract 3-8 domain keywords useful for research queries. These should be terms that would help find relevant papers and Kaggle notebooks.

Examples:
- "vehicle health prediction using sensor readings" → `["vehicle health", "predictive maintenance", "sensor data", "condition monitoring"]`
- "medical diagnosis from blood test results" → `["medical diagnosis", "clinical laboratory", "blood biomarkers", "diagnostic classification"]`
- "customer churn prediction for SaaS" → `["customer churn", "SaaS", "retention", "subscription"]`

### Step 4: Write Artifact

Write the context artifact as JSON using the Write tool.

**Artifact path**: `{DATA_DIR}/.gtd-context-{data_filename_without_extension}.json`
Example: for `~/data/titanic.csv` → `~/data/.gtd-context-titanic.json`

```json
{
  "data_path": "<absolute path to the data file>",
  "timestamp": "<current ISO 8601 timestamp>",
  "domain": "<synthesized domain description from user input + follow-ups>",
  "user_input": "<original free-text context, verbatim>",
  "follow_up_answers": {"<question>": "<answer>", ...},
  "keywords": ["<extracted domain keywords for research queries>"]
}
```

### Step 5: Confirm

Print:

```
## Context Saved

Domain: {domain}
Keywords: {keywords joined by ", "}
Artifact: {artifact_path}

This context will be used by `/gtd:eda` and `/gtd:train` to improve analysis and research queries.
```
