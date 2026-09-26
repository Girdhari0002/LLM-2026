# My LLM

A small GPT-style language model built with PyTorch and exposed through a
FastAPI backend. The API supports text generation, model training, health
checks, and automatic deployment on Vercel.

## Project structure

```text
my-llm/
├── api/
│   └── index.py          # Vercel entrypoint
├── python-api/
│   └── main.py           # FastAPI application
├── model.py              # GPT model implementation
├── train.py              # Model training logic
├── dataset.py            # Training data helpers
├── tokenizer.py          # Vocabulary and tokenization
├── generate.py           # Local text generation script
├── model.pth             # Trained model weights
├── vocab.json            # Model vocabulary
├── requirements.txt      # Python dependencies
└── vercel.json           # Vercel configuration
```

## Requirements

- Python 3.10 or newer
- PyTorch
- FastAPI
- A trained `model.pth` file
- A `vocab.json` file

## Run locally

From the project directory:

```powershell
cd C:\Users\girdh\GPT2026\my-llm
```

Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Start the FastAPI server:

```powershell
uvicorn python-api.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

## API endpoints

### Health check

```http
GET /health
```

Example:

```powershell
curl http://127.0.0.1:8000/health
```

### Generate text

```http
POST /generate
```

Example:

```powershell
curl -X POST http://127.0.0.1:8000/generate `
  -H "Content-Type: application/json" `
  -d '{\"prompt\":\"The\",\"maxTokens\":20}'
```

Request fields:

| Field | Type | Default | Description |
|---|---|---:|---|
| `prompt` | string | required | Starting text |
| `temperature` | number | `0.7` | Sampling randomness |
| `maxTokens` | integer | `50` | Maximum generated tokens |
| `topK` | integer | `40` | Top-k sampling |
| `topP` | number | `0.9` | Nucleus sampling |
| `repetitionPenalty` | number | `1.15` | Reduces repeated tokens |

### Train the model

```http
POST /train
```

Example request body:

```json
{
  "text": "Training text goes here.",
  "steps": 50000,
  "learningRate": 0.0003
}
```

Training is intended for local or persistent server environments. Vercel
functions are stateless, so files written during a Vercel request should not be
treated as permanent model storage.

## Deploy on Vercel

This project includes [vercel.json](./vercel.json) and the Vercel Python
entrypoint at [api/index.py](./api/index.py).

### GitHub and Vercel dashboard method

1. Create a new repository on GitHub.
2. Push this project to GitHub:

   ```powershell
   cd C:\Users\girdh\GPT2026\my-llm
   git init
   git add .
   git commit -m "Prepare project for Vercel deployment"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/my-llm.git
   git push -u origin main
   ```

3. Open <https://vercel.com/new>.
4. Sign in with GitHub and import the `my-llm` repository.
5. Keep the framework preset as **Other**.
6. Leave build and output directory settings at their defaults.
7. Click **Deploy**.
8. Test the deployed API:

   ```text
   https://YOUR-PROJECT.vercel.app/health
   ```

More deployment notes are available in [DEPLOYMENT.md](./DEPLOYMENT.md).

## Deployment limitations

PyTorch and the model weights can make the Vercel function large and memory
intensive. Vercel may reject the deployment or the function may exceed its
runtime limits. If that happens, deploy the FastAPI backend on Render, Railway,
or another persistent Python service, and use Vercel for a frontend if needed.

The `venv/`, `best_model.pth`, `checkpoint.pth`, and `.vercel/` directories are
excluded through [.gitignore](./.gitignore).

## License

Add the project's license here before publishing it publicly.
