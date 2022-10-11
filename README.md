# speaker-identification
Code and data for Zhou et al. "Cross-Lingual Speaker Identification Using Distant Supervision", Arxiv 2022

## Data

### Distant Supervision
We provide the distant supervision data as reported: `data/si_distant_55k.csv`

### Speaker Identification Data
We provide the datasets (train/test) that we used for experiments under `data`

## Code

### Distant Supervision Extraction (WIP)
Our distant supervision extraction pipeline is provided under `code/extraction`. 
The current code we provide is for reference, and we will further clean the code into more usable packing.
`gutenberg_pp_style_sample.txt` is a sample data from the Gutenberg project, in a chapter-by-chapter format. 

To run the extraction, start with the `run_gutenberg_coref()` function in `extractor_distant.py`, which will run a coreference model on the Gutenberg data.
Then, `run_coref_gutenberg.py` provides a `parse_overlap()` function that runs rule-based speaker identification based on the coreference results, according to Section3 of the paper. 
Finally, `format_to_roberta()` in `extractor_distant.py` formats the results into Roberta modeling inputs and labels.

### Experiments (WIP)

The training script is provided as `code/experiments/run.sh`. We will update evaluation scripts soon.
