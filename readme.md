# Movie Review Sentiment Classifier

This project fine-tunes **distilbert-base-uncased** with an 8k row
dataset [Rotten Tomatoes movie review dataset](https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes)
.The model predicts whether a movie review is **positive** or **negative**.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/changooman/analyze-reviews

2. Navigate to the Project Directory:

   ```cd analyze-reviews```

3. Install Dependencies:

   ```pip install -r requirements.txt```

## Optional

If you have a CUDA compatible GPU and want to use it for running this script

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

5. Run the script:

   ```python analyze_reviews.py```

## Example Output

```
Test accuracy: 84.9 %
INFO | Tokenizer trained found.
INFO | Pipeline built.
That movie was horrible! --> {'label': 'LABEL_0', 'score': 0.9999457597732544}
I loved that movie. --> {'label': 'LABEL_1', 'score': 0.9998169541358948}
```