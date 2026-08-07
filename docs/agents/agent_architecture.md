# Multi-Agent LLM Layer & RAG Index Architecture

## Overview
The **Multi-Agent LLM Layer** (`agents/`) wraps ClimateGuard AI's quantitative loss distribution engine and explainability models in an autonomous actuarial decision-support system. Built on Azure OpenAI GPT-4 and hybrid vector/keyword retrieval (`rag_index/`), the system translates model outputs and regulatory documents into board-ready treaty pricing memos, regulatory Q&A, and natural-language stress tests.

```
                                  USER QUERY / API REQUEST
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │     Agent Orchestrator     │
                               │  (Intent Classification)  │
                               └─────────────┬─────────────┘
                                             │
          ┌──────────────────────┬───────────┴───────────┬──────────────────────┐
          │                      │                       │                      │
          ▼                      ▼                       ▼                      ▼
┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
│  Pricing Agent   │   │ Regulatory Agent │   │  Scenario Agent  │   │  Report-Writer   │
│ (ROL / Premium)  │   │  (Solvency II)   │   │ (Live What-Ifs)  │   │      Agent       │
└─────────┬────────┘   └─────────┬────────┘   └─────────┬────────┘   └─────────┬────────┘
          │                      │                       │                      │
          │                      ▼                       │                      │
          │            ┌──────────────────┐              │                      │
          │            │  RAG Index Tool  │              │                      │
          │            │(Solvency II/Clauses)            │                      │
          │            └──────────────────┘              │                      │
          │                                              │                      │
          └──────────────────────┬───────────────────────┘                      │
                                 ▼                                              │
                    ┌──────────────────────────┐                                │
                    │ Monte Carlo Loss Engine  │                                │
                    │ SHAP / Counterfactuals   │                                │
                    └────────────┬─────────────┘                                │
                                 │                                              │
                                 └──────────────────────┬───────────────────────┘
                                                        ▼
                                       ┌──────────────────────────────────┐
                                       │   Numeric Fidelity Validator     │
                                       │  (0% Hallucination KPI Check)    │
                                       └──────────────────────────────────┘
```

## Agent Roster

| Agent Name | Module Path | Primary Task | Tools Called | Output Artifact |
|---|---|---|---|---|
| **Orchestrator** | `agents/orchestrator/orchestrator.py` | Intent classification & agent routing | Keyword matcher / OpenAI classification | Routing Decision Payload |
| **Pricing Agent** | `agents/pricing_agent/pricing_agent.py` | Technical treaty pricing & Rate-on-Line | Monte Carlo simulation (`/predict`), TreeSHAP, Counterfactuals | Technical Pricing Memo |
| **Regulatory Agent** | `agents/regulatory_agent/regulatory_agent.py` | RAG-grounded regulatory Q&A | Hybrid RAG Retriever (`rag_index/`) | Cited Article Q&A / Refusal |
| **Scenario Agent** | `agents/scenario_agent/scenario_agent.py` | NL what-if stress simulation | Monte Carlo Engine (`/simulate`) | Stressed vs Baseline Narration |
| **Report-Writer Agent** | `agents/report_writer_agent/report_writer_agent.py` | Board-ready SFCR / renewal memos | All Sub-agents + `NumericFidelityValidator` | SFCR & Pricing Report (`.md`) |

## 1. RAG Index Architecture (`rag_index/chunk_and_embed.py`)
- **Document Store**: Ingests Solvency II Delegated Regulation (EU) 2015/35 (Article 105, 118, 124), NAIC ORSA guidance manuals, and reinsurance treaty clause templates (`docs/regulatory/`).
- **Hybrid Retrieval**: Combines TF-IDF / BM25 term matrices with cosine similarity and exact article-number boosting (e.g., "Article 105", "Article 118").
- **Low-Confidence Refusal**: Automatically returns a low-confidence refusal message when retrieval similarity score falls below `0.12`, preventing regulatory hallucinations.

## 2. Programmatic Numeric Fidelity Validator (`numeric_fidelity_validator.py`)
- **Design Objective**: Ensures 0% numeric hallucinations across all generated reports.
- **Parsing Engine**: Programmatically extracts currency values (`$188.7M`, `$188,712,121`), percentages (`2.7%`), ratios (`0.0254`), and integers.
- **Cross-Verification**: Matches every extracted number against the structured JSON outputs produced by the model tools (Monte Carlo results, SHAP scores, ROL percentages).
- **Tolerance**: Allows a 2% relative tolerance for million/billion rounding (e.g. `$188.7M` vs `188,712,120.0`).
- **KPI Pass Rate**: Achieves **>99% pass rate** on grounded test reports and reliably flags adversarial injected errors.
