# News Sentiment Analysis

A professional end-to-end financial news sentiment analysis project. Analyzes financial news headlines, computes sentiment scores, calculates technical indicators from stock price data, and explores correlations between sentiment and stock returns.

## Project Structure

```
news-sentiment-analysis/
├── src/                # Reusable Python modules
│   ├── data_loader.py      # Data acquisition and loading
│   ├── sentiment.py        # Sentiment analysis (VADER, TextBlob)
│   ├── indicators.py       # Technical indicators computation
│   ├── visualization.py    # Publication-quality charts
│   └── utils.py            # Shared utility functions
├── notebooks/          # Jupyter notebooks
│   ├── task_1_eda.ipynb            # Exploratory data analysis
│   ├── task_2_technical_analysis.ipynb  # Technical indicators
│   └── task_3_sentiment_correlation.ipynb # Sentiment vs returns
├── tests/             # Unit tests
├── scripts/           # Helper utility scripts
├── data/raw/          # Raw input data (not tracked)
├── .github/workflows/ # CI/CD pipeline
└── requirements.txt   # Python dependencies
```

## Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/fenet-as/news-sentiment-analysis.git
   cd news-sentiment-analysis
   ```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data**

   ```python
   python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt')"
   ```

## Usage

### Run the notebooks

Start Jupyter Lab and open any notebook in the `notebooks/` directory:

```bash
jupyter lab
```

### Run tests

```bash
python -m pytest tests/ -v
```

### Download sample data

```bash
python scripts/download_data.py
```

## Modules Overview

| Module | Description |
|--------|-------------|
| `src.data_loader` | Fetch stock prices via `yfinance`, load CSV/JSON datasets |
| `src.sentiment` | Compute VADER and TextBlob sentiment scores from text |
| `src.indicators` | Calculate SMA, EMA, RSI, MACD using `pandas` |
| `src.visualization` | Create publication-quality `matplotlib`/`seaborn` charts |
| `src.utils` | Shared helpers for date parsing, logging, config |

## Dependencies

- pandas, numpy — data manipulation
- matplotlib, seaborn — visualization
- nltk, vaderSentiment, textblob — NLP sentiment
- scikit-learn — correlation and regression
- yfinance — stock price data
- jupyter — interactive notebooks
- pytest — testing

## CI/CD

The project uses GitHub Actions for continuous integration. On every push to `main` or `task-*` branches, the workflow:

1. Installs dependencies
2. Runs flake8 linting
3. Executes pytest test suite

## License

MIT
