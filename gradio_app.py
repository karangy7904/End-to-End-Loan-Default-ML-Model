"""
Gradio demo UI for the loan risk model — reuses the exact same
model_service.predict() function as the FastAPI app, so predictions are
always consistent between the API and this UI.

Run locally:  python gradio_app.py       -> http://localhost:7860
"""
import gradio as gr
import pandas as pd
import os

from app.schemas import LoanApplication
from app import model_service
from src import config

HOUSE_OWNERSHIP = ["rented", "owned", "norent_noown"]
CAR_OWNERSHIP = ["no", "yes"]
MARITAL_STATUS = ["single", "married"]

# Profession / CITY / STATE have 51 / 317 / 29 distinct values — pull the
# real choices from the training data so the dropdowns always match what
# the model was trained on. Falls back to a small illustrative list if
# the training CSV is not present.
try:
    _train_df = pd.read_csv(config.TRAIN_PATH)
    PROFESSIONS = sorted(_train_df["Profession"].unique().tolist())
    CITIES = sorted(_train_df["CITY"].unique().tolist())
    STATES = sorted(_train_df["STATE"].unique().tolist())
except FileNotFoundError:
    PROFESSIONS = ["Software_Developer", "Mechanical_engineer", "Analyst"]
    CITIES = ["Rewa", "Parbhani"]
    STATES = ["Madhya_Pradesh", "Maharashtra"]


def score_application(
    income, age, experience, married_single, house_ownership, car_ownership,
    profession, city, state, current_job_yrs, current_house_yrs,
):
    try:
        application = LoanApplication(
            income=income, age=int(age), experience=int(experience),
            married_single=married_single, house_ownership=house_ownership,
            car_ownership=car_ownership, profession=profession, city=city,
            state=state, current_job_yrs=int(current_job_yrs),
            current_house_yrs=int(current_house_yrs),
        )
        result = model_service.predict(application)
    except model_service.ModelNotAvailableError as e:
        return f"⚠️ {e}", "", ""
    except Exception as e:
        return f"⚠️ Error: {e}", "", ""

    prob_pct = f"{result['risk_probability'] * 100:.1f}%"
    decision = result["decision"]
    decision_emoji = "✅" if decision == "APPROVE" else "🔎"
    return (
        f"{prob_pct} probability of high risk",
        f"Risk score: {result['risk_score']} / 1000",
        f"{decision_emoji} Decision: {decision} (threshold={result['threshold_used']:.2f})",
    )


with gr.Blocks(title="Loan Risk Predictor") as demo:
    gr.Markdown("# 🏦 Loan Risk Predictor\nEnter applicant details to estimate default/risk probability.")

    with gr.Row():
        with gr.Column():
            income = gr.Number(label="Annual income", value=1300000)
            age = gr.Slider(18, 100, value=30, step=1, label="Age")
            experience = gr.Slider(0, 40, value=5, step=1, label="Total work experience (years)")
            married_single = gr.Dropdown(MARITAL_STATUS, label="Marital status", value="single")
        with gr.Column():
            house_ownership = gr.Dropdown(HOUSE_OWNERSHIP, label="House ownership", value="rented")
            car_ownership = gr.Dropdown(CAR_OWNERSHIP, label="Car ownership", value="no")
            profession = gr.Dropdown(PROFESSIONS, label="Profession", value=PROFESSIONS[0])
            current_job_yrs = gr.Slider(0, 40, value=3, step=1, label="Years at current job")
        with gr.Column():
            city = gr.Dropdown(CITIES, label="City", value=CITIES[0])
            state = gr.Dropdown(STATES, label="State", value=STATES[0])
            current_house_yrs = gr.Slider(0, 40, value=10, step=1, label="Years at current residence")

    predict_btn = gr.Button("Predict risk", variant="primary")

    with gr.Row():
        out_prob = gr.Label(label="Risk probability")
        out_score = gr.Label(label="Risk score")
        out_decision = gr.Label(label="Decision")

    predict_btn.click(
        fn=score_application,
        inputs=[
            income, age, experience, married_single, house_ownership,
            car_ownership, profession, city, state, current_job_yrs,
            current_house_yrs,
        ],
        outputs=[out_prob, out_score, out_decision],
    )

if __name__ == "__main__":
    demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860)),
)
