"""Configuration loader: merges YAML config with .env secrets."""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file and environment variables."""
    project_root = Path(__file__).parent.parent
    load_dotenv(project_root / ".env")

    yaml_path = project_root / config_path
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    # Override mode from env
    cfg["mode"] = os.getenv("MODE", cfg.get("mode", "paper"))

    # Inject secrets
    cfg["kalshi"]["api_key_id"] = os.getenv("KALSHI_API_KEY_ID", "")
    cfg["kalshi"]["private_key_path"] = os.getenv("KALSHI_PRIVATE_KEY_PATH", "")

    # Override base URL for live mode
    if cfg["mode"] == "live":
        cfg["kalshi"]["base_url"] = cfg["kalshi"]["live_url"]

    env_url = os.getenv("KALSHI_BASE_URL")
    if env_url:
        cfg["kalshi"]["base_url"] = env_url

    # Discord webhook from env (overrides config.yaml)
    discord_url = os.getenv("DISCORD_WEBHOOK_URL")
    if discord_url:
        cfg.setdefault("discord", {})["webhook_url"] = discord_url

    # Ensure data directory exists
    db_path = project_root / cfg["database"]["path"]
    db_path.parent.mkdir(parents=True, exist_ok=True)
    cfg["database"]["path"] = str(db_path)

    csv_path = project_root / cfg["logging"]["trade_csv"]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    cfg["logging"]["trade_csv"] = str(csv_path)

    return cfg
