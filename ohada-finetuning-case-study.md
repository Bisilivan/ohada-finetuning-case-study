# Zero Out of Forty: What Fine-Tuning Llama 3.2 3B on OHADA Commercial Law Actually Taught It

*A case study in legal dataset construction, evaluation methodology, and a negative result worth reporting.*

## The gap

OHADA, the Organisation for the Harmonization of Business Law in Africa, governs commercial law across 17 countries and over 200 million people. It is a real, codified, harmonized legal system, cited by real courts, including a supranational court of cassation (the CCJA), backed by a substantial body of jurisprudence and doctrine. It also appears to be substantially underrepresented in the training data of general-purpose LLMs, though the extent of that underrepresentation is hard to measure directly, and this report does not attempt to. What this report does measure directly: a small, general-purpose 3B model, asked OHADA-specific commercial law questions, answers fluently, cites an article number with total confidence, and is wrong far more often than not. Whether that holds for larger or more capable models is a separate question this experiment was not designed to answer.

I am a practicing OHADA business lawyer with 14 years of legal experience (Barreau du Cameroun since 2020). Over the past several months I built a dataset to test, empirically, whether that gap could be narrowed, and, more importantly, whether fine-tuning is actually the right tool to narrow it. This is a report on what I found, including the part that didn't work.

## The dataset

The training set: 114 examples in Alpaca/JSONL format covering AUDCG (the Uniform Act on General Commercial Law, one of ten OHADA Uniform Acts), spanning case-qualification questions, direct Q&A, and defense-brief-style arguments across four difficulty tiers. A 10-entry public sample, same schema, same sourcing standard, is on the Hub at [Bisilivan/dataset-ohada-droit-commercial-general-echantillon](https://huggingface.co/datasets/Bisilivan/dataset-ohada-droit-commercial-general-echantillon). Each example is built under a strict sourcing hierarchy: the Uniform Act text first, then CCJA jurisprudence, then named doctrine (Pougoué's *Encyclopédie du droit OHADA*, Kuate Tamèghe, and others), never an unsourced assertion.

A held-out test set of 40 items has zero literal overlap with the training set, verified programmatically before training began. Its composition:

| By question type | n | | By difficulty | n |
|---|---:|---|---|---:|
| Case qualification | 24 | | Basic | 7 |
| Direct Q&A | 11 | | Intermediate | 17 |
| Defense-brief argument | 5 | | Advanced | 12 |
| | | | Expert | 4 |
| **Total** | **40** | | **Total** | **40** |

Forty items are enough to make a result like 0/40 on citation accuracy hard to dismiss as noise; what that sample size does and doesn't support is addressed in Limitations, below.

## The experiment

A note on authorship, since it matters for how to read the "I" in what follows. I am a lawyer, not an ML engineer. I specified what the experiment needed to prove and to what standard: a sourcing discipline for every training example, a genuinely held-out test set with zero overlap, deterministic generation so the two models are compared on equal footing, and a two-tier evaluation, detailed below. An AI assistant wrote and debugged the training and evaluation code, ran the pipeline, and performed the LLM-judge pass under that specification, with me reviewing, correcting, and pushing back at each step, including on an earlier, overclaiming draft of this report's own conclusion. The legal content, the sourcing standards, and the judgment calls on what the results mean are mine. The code is not, and I am not going to pretend otherwise.

The setup: Llama-3.2-3B-Instruct, LoRA (r=16, alpha=32, targeting all attention and MLP projection matrices), 114 training examples, 3 epochs, a single T4 GPU (Kaggle's free tier). Training loss dropped cleanly from 1.90 to 1.19, a well-behaved run by every conventional signal.

Generation followed the same discipline: 40 paired responses to the 40 held-out questions, one from the untouched base model, one from the fine-tuned model, identical deterministic greedy decoding for both, so the only variable between the two answer sets is the training itself.

## The evaluation

Rather than eyeball 80 answers and call it a day, I built a two-tier evaluation: an LLM-judge pass across all 40 items, scored on five criteria (0 to 2 each): legal qualification, citation accuracy, reasoning, application to the facts, and absence of hallucination, followed by a curated subset going to an external OHADA-practicing colleague, specifically because the dataset's author cannot be a neutral evaluator of his own work.

Precision on what the LLM-judge pass actually was, since it changes how much weight it should carry: a single LLM, given each question, its pre-validated reference answer, and both the base and fine-tuned responses side by side, scored the five criteria with a short justification per item. It was not blind: the judge knew which answer came from which model on every item, and items were read in a fixed rather than randomized order. It was a single analytical pass, not a fixed prompt run through an API at a set temperature, so it is not mechanically reproducible in the way an automated benchmark would be. Citation accuracy was scored by checking whether the article, case, or doctrinal source cited matched what the reference answer relies on, not by independently re-verifying the underlying legal text against the Uniform Acts. These are real constraints on what this tier can claim, not a footnote to wave past: it is a structured, evidence-linked first screen, not a blinded or reproducible benchmark. Authority on legal accuracy specifically rests with the second tier, the external practitioner review, which was not yet complete at the time of writing.

## The result

Citation accuracy: **0 out of 40, on both models, before and after fine-tuning.**

Not one of the 80 answers cited an article, a CCJA decision, or a doctrinal source that actually matched what the held-out reference answer contains. Both models produce fluent, confidently formatted legal French, correct citation syntax, properly nested quotation marks, doctrine references in the right house style, wrapped around content that is, overwhelmingly, fabricated. The fine-tuned model did not get better at this. On one representative question, the base model invents a citation to a "Convention de Rome" (an EU private-international-law instrument with no OHADA existence), while the fine-tuned model, on the identical question, opens coherently and then spirals into a compounding, self-referential enumeration of nonexistent legal terms until it hits the token limit.

Verdict-level accuracy, whether the model landed on the legally correct conclusion, was roughly comparable between the two, a gap too small to treat as significant on n=40. Reasoning quality and freedom from hallucination were, if anything, marginally worse after fine-tuning. Degenerate repetition loops, the model getting stuck compounding the same phrase until it hits the token limit, were more frequent post fine-tuning.

| Metric | Base | Fine-tuned | Δ |
|---|---:|---:|---:|
| Correct legal verdict | 14/40 (35%) | 16/40 (40%) | +5 pp |
| Citation accuracy | 0/40 (0%) | 0/40 (0%) | 0 |
| Reasoning (avg, /2) | 0.10 | 0.03 | −0.07 |
| Application to facts (avg, /2) | 0.10 | 0.07 | −0.03 |
| Hallucination-free (avg, /2) | 0.03 | 0.00 | −0.03 |
| Degenerate repetition loops | 4/40 (10%) | 9/40 (23%) | +13 pp |

The 5-point verdict-accuracy gap is two additional questions answered correctly, out of forty. I would not build an argument on it either way. The 0/40 on citation accuracy and the near-doubling of repetition loops are the two numbers in this table I'd actually defend as signal rather than noise.

## What this means

Training loss went down. Legal accuracy did not go up. Those two facts, side by side, are the actual finding.

This tracks a result already documented in the literature (Gekhman et al., EMNLP 2024): fine-tuning a model on new factual knowledge tends to increase its propensity to hallucinate confidently once that knowledge is only partially absorbed, because the model learns the surface pattern of a well-sourced answer (structure, tone, citation syntax) far more readily than it learns the specific fact that belongs in each citation slot. A phrasing pattern recurs across dozens of unrelated training examples; a specific article number appears only in the handful of examples that happen to concern that article. Gradient descent has far more signal to learn the former than the latter, especially from 114 examples on a 3B-parameter model.

The practical implication is narrower than "fine-tuning doesn't work," and it deserves to be stated precisely: what this experiment rules out is LoRA-based supervised fine-tuning of an instruction-tuned 3B model on 114 reasoning examples, nothing more. Two adjacent hypotheses remain untested and deserve to be named rather than skipped past.

First, full-parameter fine-tuning trades LoRA's efficiency for materially more representational capacity, and the literature on knowledge-intensive fine-tuning is genuinely split on whether that difference matters for exactly this kind of high-entropy, individually verifiable factual injection. Biderman et al. ("LoRA Learns Less and Forgets Less," 2024) found LoRA under-performing full fine-tuning specifically on knowledge-intensive tasks. Whether it would have helped here is an open question, not a settled one, and it comes with a cost this project already flagged once before: more trainable capacity on only 114 examples raises the risk of memorizing the training set itself rather than generalizing to unseen questions, exactly the failure mode a held-out test set and a two-tier evaluation exist to catch.

Second, continued pretraining on a much larger corpus of raw OHADA legal text (the Uniform Acts and CCJA decisions themselves, not this reasoning-demonstration dataset, and at a scale several orders of magnitude beyond 114 examples) is a different regime entirely. For scale: the training set totals roughly 138,000 tokens (114 examples at an average tokenized sequence length of 1,215 tokens, question, system prompt, and answer combined, measured directly before training), seen three times over across the three training epochs. Meaningful continued pretraining for domain adaptation typically operates on corpora several orders of magnitude larger, commonly cited in the hundreds of millions of tokens. This experiment says nothing about that regime either way.

What the experiment does establish is a lower bound, not a general verdict on fine-tuning. Retrieval-augmented generation remains, in my view, the most reliable fix, because it sidesteps the need for the model to memorize the fact at all, rather than because the alternatives above have been tried and shown to fail. The dataset itself, built under a strict, auditable sourcing discipline, remains useful regardless of which downstream method eventually consumes it: as RAG grounding data, as an evaluation benchmark, or as training data for one of the untested regimes above.

## Limitations

Worth naming three limitations not already covered above. Forty held-out items make a result like 0/40 hard to dismiss, but they are not enough to treat a five-point verdict-accuracy gap as meaningful, or to break results down by difficulty tier or question type without small-sample noise. The dataset covers AUDCG, one of ten OHADA Uniform Acts; nothing here speaks to how the same method would fare on company law, secured transactions, or the others. And the result is specific to Llama-3.2-3B-Instruct with this exact LoRA configuration; a larger base model, a different rank, or a different set of target modules could behave differently, none of which this report tested.

## Why this is worth reading if you evaluate model outputs for a living

I am sharing the negative result on purpose. The methodology, a held-out test set with zero overlap, deterministic generation, a two-tier evaluation separating a scalable but non-authoritative LLM pass from an authoritative human domain expert, and a five-criteria rubric that scores citation accuracy separately from surface fluency, is the part I would want evaluated, not the fine-tune itself. A model that sounds right and a model that is right diverge exactly where most surface-level QA never looks. OHADA commercial law, underrepresented in training data and unusually citation-dense, makes that divergence easy to catch and hard to ignore.

The full test set is in this repository's `test-set/` folder: the 40 reference answers (`references.jsonl`), the base and fine-tuned model responses to each question (`reponses_temoin.jsonl`, `reponses_finetune.jsonl`), and a complete description of the training protocol (`protocole-entrainement.md`).

---

*Ivan Bertrand Ngomen Bisil, OHADA business lawyer, Barreau du Cameroun. Building legal AI training data at the intersection of legal practice and dataset engineering. huggingface.co/Bisilivan*
