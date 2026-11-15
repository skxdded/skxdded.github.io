# Brainoware Reservoir Lab

An interactive rebuild of the computing method behind **Brainoware** (Cai et al., *Nature Electronics*, 2023): a fixed random "reservoir" of neurons plus a trained linear readout, applied to speaker recognition on the UCI Japanese Vowels dataset.

**Live site:** https://skxdded.github.io/

The reservoir is **simulated**. No living neurons are involved, and the dataset differs from the paper's, so my accuracies cannot be compared with Brainoware's ~78%. What I rebuilt is the method, not the result.

## What's in this repo

```
skxdded.github.io/
├─ index.html            the website
├─ css/style.css
├─ js/
│  ├─ core.js            the experiment in JavaScript: reservoir + readout
│  ├─ worker.js          runs it off the main thread
│  └─ app.js             terminal, charts, inspector
├─ data/                 ae.train and ae.test, loaded by the website
├─ experiment/           clean copy of the Python version
│  ├─ reservoir_vowels.py
│  └─ requirements.txt
└─ THEBUILD/             my original build: the run behind my article
   ├─ brainowareBuild/   the Python script I ran
   ├─ data/              dataset download used by the Python script
   ├─ results.csv        results from the last run
   └─ accuracy_vs_size.png
```

## Use the website
Open https://skxdded.github.io/ and type `help` in the terminal. Try `run`, `baseline`, `noise`, or change one setting, for example `set radius 0.5`, then `run` again. Everything runs on your own device and nothing is uploaded.

To run it locally:
```
python -m http.server
# open http://localhost:8000
```

## Run the Python version
```
pip install numpy scikit-learn matplotlib
python THEBUILD\brainowareBuild\reservoir_vowels.py --noise
```
The script downloads the dataset on its own the first time. It saves `results.csv` and `accuracy_vs_size.png` into the folder you ran it from.

## My results (Python, 5 random reservoirs per size, 270 training / 370 test recordings)

Baseline with no reservoir: **97.8%**

| Reservoir size | Spectral radius 0.9 | Spectral radius 0.5 |
|---|---|---|
| 10 neurons | 91.9% ± 3.8% | 92.7% ± 3.4% |
| 50 neurons | 98.3% ± 0.3% | 98.8% ± 0.4% |
| 100 neurons | 99.0% ± 0.3% | 99.1% ± 0.1% |
| 300 neurons | 99.2% ± 0.2% | 99.0% ± 0.4% |
| 1000 neurons | 99.1% ± 0.3% | 99.3% ± 0.1% |

The reservoir was slightly but consistently better than the baseline from 50 neurons up. Changing the spectral radius from 0.9 to 0.5 made little or no real difference, since most gaps are inside the error bars. This task is easy (a plain classifier already scores 97.8%), so these results show the method works, not that the reservoir is much better.

## How the website's version was checked
The reservoir rescaling, the eigen-solver and the ridge classifier in `js/core.js` were each compared against numpy and scikit-learn on the same inputs, and they agreed on the chosen regularisation strength and test accuracy. The website draws different random reservoirs than the Python script, so results are close but not identical.

## Limits
- No living neurons and no electrode array.
- A different dataset from the paper's.
- One dataset, one task, one test split.
- Brainoware's organoid changes as it is stimulated; my reservoir is fixed.

## Credits
- Cai et al. (2023), *Brain organoid reservoir computing for artificial intelligence*, Nature Electronics. https://doi.org/10.1038/s41928-023-01069-w
- Lukoševičius & Jaeger (2009), *Reservoir computing approaches to recurrent neural network training*.
- UCI Machine Learning Repository, Japanese Vowels. https://doi.org/10.24432/C5NS47
- I used Claude, an AI assistant, to help write the code. I ran, debugged and tested it myself.

By Moksha Chaudhari, Grade 11, Sinclair Secondary (TKS).