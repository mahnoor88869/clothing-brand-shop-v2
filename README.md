# Clothing Shop Sales and Customer Feedback Assistant

This repository implements **AI for Everyone Project 02** using the supplied `Project_02` dataset pack.

## Important
- The sales table is generated classroom data, not real shop data. Predictions are demonstration estimates, not confirmed sales forecasts.
- The clustering groups are product-planning scenarios, not customer segments.
- The image classifier supports only `top`, `trouser`, and `bag` and may not generalise to phone photos or multiple garments.
- The app loads saved models only. It does not train models when a user clicks a button.
- The app works without a Groq API key using the Offline summary. Groq is optional and must remain on its free plan.

## Required project layout
```text
app.py
requirements.txt
README.md
models/
  supervised.joblib
  clusters.joblib
  images.keras
  group_descriptions.json
data/
  main.csv
  feedback.csv
  example_inputs.csv
  meta.json
  topic_keywords.json
  prompt_test_cases.json (if supplied)
sample_images/
nltk_data/
source_notices/
```

The training-only `data/images.npz` is deliberately not needed in the public app repository. Keep it in Colab or private storage. Retain `source_notices/`. Never upload `.streamlit/secrets.toml` or an API key.

## Train and build
1. Open `notebooks/Clothing_Shop_Full_Workflow.ipynb` in Google Colab.
2. Upload the instructor-supplied `AI_for_Everyone_Ready_Datasets.zip` when prompted.
3. Run cells from top to bottom. The notebook uses the fixed supplied splits, trains Decision Tree, KMeans, the specified 3-epoch CNN, evaluates VADER, and saves model artifacts.
4. Download the deployment ZIP created by the final notebook cell.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Optional Groq Secrets
In Streamlit Community Cloud → App → Settings → Secrets, add:
```toml
GROQ_API_KEY = "your-key"
GROQ_MODEL = "openai/gpt-oss-20b"
```
Check the currently available model and free quota in your Groq account. Do not enable paid billing.

## Deployment
Push the deployment folder to a GitHub repository, then connect it in Streamlit Community Cloud and select `app.py`. Use Python and dependency versions compatible with the TensorFlow version used to train `images.keras`.
