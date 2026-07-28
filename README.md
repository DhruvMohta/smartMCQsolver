# Smart MCQ Solver Challenge

Course project for Deep Learning and Generative AI, Diploma Level.

Roll number: 22f3000114

## The problem

Given a science multiple choice question with five options A to E, predict the three most likely
answers in ranked order. Scored with MAP@3, so the correct answer earns full credit at rank one,
half at rank two, and a third at rank three.

## Repository layout

The work was done milestone by milestone. Each milestone has its own branch, and `main` holds the
final notebook that brings everything together.

| Branch | Contents |
|---|---|
| `milestone-1` | Data exploration and first baselines |
| `milestone-2` | Similarity baselines using TF-IDF and Word2Vec |
| `milestone-3` | Retrieval pipeline over Wikipedia with FAISS |
| `milestone-4` | QLoRA fine tuning of Llama 3.1 8B |
| `milestone-5` | Ensembling and MAP@3 evaluation |
| `main` | Final notebook, this is the one to look at |

Branch names may differ slightly, check the branch list on the repository page.

## The final notebook

`dl-22f3000114-notebook-t22026-final.ipynb` on `main` contains the whole pipeline in one file.

It has three models of different architectures:

1. A bidirectional LSTM built from scratch. Every weight is trained here and nothing pretrained is
   loaded into it.
2. Llama 3.1 8B, a pretrained model, used with retrieved Wikipedia context.
3. An ensemble that averages four signals from that model, base and fine tuned, each with and
   without retrieved context.

## How the notebook runs

There is one switch at the top called `BUILD`.

Set `BUILD = True` to train the from scratch model, run retrieval, query Llama for every signal,
and save all the outputs to the working directory. This is the slow run and only needs doing once.

Set `BUILD = False` to skip all of that and load the saved arrays instead. Then set `SUBMIT_MODEL`
to whichever model you want and the notebook writes `submission.csv`. This runs in under a minute
and does not need a GPU.

Because submissions read saved probability arrays rather than rerunning the model, the same
selection always produces the same file.

## Running it on Kaggle

Attach these inputs before running:

1. The competition data
2. `jjinho/wikipedia-2023-07-faiss-index`
3. `jjinho/wikipedia-20230701`

Set the accelerator to GPU T4 x2. Add your Weights and Biases key as a Kaggle secret named `WANDB`,
and a Hugging Face token named `HF_TOKEN`.

## A note on the data

The training file has 2000 rows but only around 1125 distinct questions. The same question appears
several times wrapped in different instruction phrases, for example "Select the most accurate
option" and "Determine the correct option" in front of identical question text.

This matters for validation. Splitting rows independently puts one phrasing of a question in
training and another phrasing of the same question in validation, so the model scores well on
questions it has already seen. The notebook strips the instruction phrase, groups on the remaining
question text, and asserts that no question appears on both sides of the split.

## Experiment tracking

All runs are logged to Weights and Biases under the project `dl-genai-project`.
