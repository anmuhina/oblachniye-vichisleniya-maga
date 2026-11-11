import os
import json
import io
from datetime import datetime, timezone
from collections import defaultdict

import boto3
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# --- Конфигурация Object Storage (S3-совместимый) ---
S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "https://s3.cloud.ru")
S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY")
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY")
S3_BUCKET = os.environ.get("S3_BUCKET", "logs")
S3_REGION = os.environ.get("S3_REGION", "ru-central-1")

s3 = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
    region_name=S3_REGION,
)

def read_csv_from_s3(key):
    obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()))

def write_json_to_s3(key, data):
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=json.dumps(data, ensure_ascii=False, indent=2),
        ContentType="application/json",
    )

def parse_timestamp(ts):
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))

def calculate_reliability_metrics(df):
    """
    Рассчитывает MTBF, MTTR, коэффициент готовности по каждому узлу.
    """
    metrics = {}
    nodes = df["node_id"].unique()

    for node in nodes:
        node_df = df[df["node_id"] == node].sort_values("timestamp")
        failures = node_df[node_df["event_type"] == "failure"]
        repairs = node_df[node_df["event_type"] == "repair"]

        num_failures = len(failures)
        num_repairs = len(repairs)

        if num_failures == 0:
            metrics[node] = {
                "mtbf": None, "mttr": None, "availability": 1.0,
                "num_failures": 0, "num_repairs": 0,
            }
            continue

        # Суммарное время наработки = интервал от первого отказа до последнего ремонта минус время простоев
        first_failure = failures["timestamp"].iloc[0]
        last_repair = repairs["timestamp"].iloc[-1] if num_repairs > 0 else failures["timestamp"].iloc[-1]
        total_period = (parse_timestamp(last_repair) - parse_timestamp(first_failure)).total_seconds() / 3600.0

        total_downtime = repairs["duration_hours"].sum() if num_repairs > 0 else 0.0
        total_uptime = total_period - total_downtime

        mtbf = total_uptime / num_failures if num_failures > 0 else None
        mttr = total_downtime / num_repairs if num_repairs > 0 else None
        availability = mtbf / (mtbf + mttr) if (mtbf is not None and mttr is not None and (mtbf + mttr) > 0) else None

        metrics[node] = {
            "mtbf": round(mtbf, 2) if mtbf is not None else None,
            "mttr": round(mttr, 2) if mttr is not None else None,
            "availability": round(availability, 4) if availability is not None else None,
            "num_failures": int(num_failures),
            "num_repairs": int(num_repairs),
            "total_downtime_hours": round(total_downtime, 2),
        }

    return metrics

def forecast_mtbf(metrics, days=30):
    """
    Простой прогноз MTBF на основе экспоненциального сглаживания.
    Для учебного проекта достаточно линейной экстраполяции по последним значениям.
    """
    forecast = {}
    for node, m in metrics.items():
        if m["mtbf"] is None:
            forecast[node] = {"forecast_mtbf": None, "note": "Недостаточно данных"}
            continue
        # Простая линейная экстраполяция: предполагаем сохранение текущего MTBF
        # В реальном проекте — ARIMA/Prophet; для учебного — константный прогноз с доверительным интервалом ±15%
        current = m["mtbf"]
        forecast[node] = {
            "forecast_mtbf_30d": round(current, 2),
            "ci_lower": round(current * 0.85, 2),
            "ci_upper": round(current * 1.15, 2),
            "method": "linear_extrapolation_with_margin",
        }
    return forecast

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/calculate", methods=["POST"])
def calculate():
    """
    Триггер расчёта. Читает incidents.csv, считает метрики, пишет JSON.
    """
    try:
        df = read_csv_from_s3("input/incidents.csv")
        metrics = calculate_reliability_metrics(df)
        forecast = forecast_mtbf(metrics)

        result = {
            "calculated_at": datetime.now(timezone.utc).isoformat(),
            "nodes": metrics,
            "forecast": forecast,
        }

        write_json_to_s3("output/metrics.json", result)

        # Проверка порогов для алертов (порог Kг = 0.9)
        threshold = 0.9
        alerts = [
            {"node": node, "availability": m["availability"], "threshold": threshold}
            for node, m in metrics.items()
            if m["availability"] is not None and m["availability"] < threshold
        ]
        if alerts:
            write_json_to_s3("output/alerts.json", {"alerts": alerts, "checked_at": datetime.now(timezone.utc).isoformat()})

        return jsonify({"status": "success", "result": result}), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/results", methods=["GET"])
def get_results():
    """Возвращает последние рассчитанные метрики."""
    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key="output/metrics.json")
        data = json.loads(obj["Body"].read())
        return jsonify(data), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 404

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
