from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import quote


@dataclass(frozen=True)
class CheckResult:
    service: str
    status: str
    details: Mapping[str, Any]


def load_env(path: Path) -> None:
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        value = value.strip().strip("\"").strip("'")
        os.environ.setdefault(name.strip(), value)


def check_configuration() -> CheckResult:
    required = (
        "VULTR_API_KEY",
        "VULTR_REGION",
        "VULTR_GPU_PLAN_ID",
        "VULTR_OS_ID",
        "VULTR_SSH_KEY_ID",
        "VULTR_SSH_PRIVATE_KEY_PATH",
        "VULTR_HOURLY_COST_USD",
        "HF_TOKEN",
        "HF_ADAPTER_REPO",
        "OPENROUTER_API_KEY",
        "OPENROUTER_FACTORY_MODEL",
        "OPENROUTER_CRITIC_MODEL",
        "JEV_API_KEY",
        "JEV_BASE_URL",
        "JEV_MODEL",
    )
    missing = [name for name in required if not os.environ.get(name)]
    return CheckResult(
        service="configuration",
        status="ok" if not missing else "blocked",
        details={
            "missing": missing,
            "vultr_region": os.environ.get("VULTR_REGION"),
            "vultr_plan": os.environ.get("VULTR_GPU_PLAN_ID"),
            "vultr_os_id": os.environ.get("VULTR_OS_ID"),
            "factory_model": os.environ.get("OPENROUTER_FACTORY_MODEL"),
            "critic_model": os.environ.get("OPENROUTER_CRITIC_MODEL"),
            "jev_model": os.environ.get("JEV_MODEL"),
        },
    )


def check_vultr() -> CheckResult:
    try:
        region = _env("VULTR_REGION")
        plan = _env("VULTR_GPU_PLAN_ID")
        ssh_key_id = _env("VULTR_SSH_KEY_ID")
        configured_hourly_cost = Decimal(_env("VULTR_HOURLY_COST_USD"))
        token = _env("VULTR_API_KEY")
        availability = _get_json(
            f"https://api.vultr.com/v2/regions/{quote(region)}/availability",
            token,
            force_ipv4=True,
        )
        keys = _get_json("https://api.vultr.com/v2/ssh-keys", token, force_ipv4=True)
        plans = _get_json(
            "https://api.vultr.com/v2/plans?per_page=500", token, force_ipv4=True
        ).get("plans", [])
        account = _get_json("https://api.vultr.com/v2/account", token, force_ipv4=True).get(
            "account", {}
        )
        instances = _get_json(
            "https://api.vultr.com/v2/instances?per_page=500", token, force_ipv4=True
        ).get("instances", [])
        available = plan in availability.get("available_plans", [])
        registered = any(item.get("id") == ssh_key_id for item in keys.get("ssh_keys", []))
        plan_record = next((item for item in plans if item.get("id") == plan), None)
        provider_hourly_cost = (
            Decimal(str(plan_record["hourly_cost"])) if plan_record is not None else None
        )
        price_matches = provider_hourly_cost == configured_hourly_cost
        status = "ok" if available and registered and plan_record and price_matches else "blocked"
        return CheckResult(
            service="vultr",
            status=status,
            details={
                "region": region,
                "plan": plan,
                "plan_available": available,
                "provider_hourly_cost_usd": str(provider_hourly_cost)
                if provider_hourly_cost is not None
                else None,
                "configured_hourly_cost_usd": str(configured_hourly_cost),
                "hourly_cost_matches": price_matches,
                "ssh_key_registered": registered,
                "registered_ssh_key_count": len(keys.get("ssh_keys", [])),
                "account_balance_usd": account.get("balance"),
                "pending_charges_usd": account.get("pending_charges"),
                "active_instance_count": len(instances),
                "selected_plan_instance_count": sum(
                    item.get("plan") == plan for item in instances
                ),
            },
        )
    except Exception as error:
        return _failed("vultr", error)


def check_huggingface() -> CheckResult:
    try:
        token = _env("HF_TOKEN")
        repo_id = _env("HF_ADAPTER_REPO")
        identity = _get_json("https://huggingface.co/api/whoami-v2", token)
        role = (
            identity.get("auth", {})
            .get("accessToken", {})
            .get("role")
        )
        permissions = []
        for scope in (
            identity.get("auth", {})
            .get("accessToken", {})
            .get("fineGrained", {})
            .get("scoped", [])
        ):
            permissions.extend(scope.get("permissions", []))
        can_write = role == "write" or "repo.write" in permissions
        repo_exists = True
        try:
            _get_json(f"https://huggingface.co/api/models/{quote(repo_id, safe='/')}", token)
        except RuntimeError as error:
            if "HTTP 404" in str(error):
                repo_exists = False
            else:
                raise
        return CheckResult(
            service="huggingface",
            status="ok" if can_write and repo_exists else "blocked",
            details={
                "user": identity.get("name"),
                "token_role": role,
                "repo_id": repo_id,
                "repo_exists": repo_exists,
                "repo_write_permission": can_write,
            },
        )
    except Exception as error:
        return _failed("huggingface", error)


def check_openrouter() -> CheckResult:
    try:
        token = _env("OPENROUTER_API_KEY")
        factory_model = _env("OPENROUTER_FACTORY_MODEL")
        critic_model = _env("OPENROUTER_CRITIC_MODEL")
        data = _get_json(
            "https://openrouter.ai/api/v1/key", token
        ).get("data", {})
        model_ids = {
            item.get("id")
            for item in _get_json("https://openrouter.ai/api/v1/models", token).get("data", [])
        }
        factory_available = factory_model in model_ids
        critic_available = critic_model in model_ids
        return CheckResult(
            service="openrouter",
            status="ok" if factory_available and critic_available else "blocked",
            details={
                "limit_usd": data.get("limit"),
                "remaining_usd": data.get("limit_remaining"),
                "usage_usd": data.get("usage"),
                "expires_at": data.get("expires_at"),
                "factory_model": factory_model,
                "factory_model_available": factory_available,
                "critic_model": critic_model,
                "critic_model_available": critic_available,
            },
        )
    except Exception as error:
        return _failed("openrouter", error)


def check_jev() -> CheckResult:
    try:
        base_url = _env("JEV_BASE_URL").rstrip("/")
        configured_model = _env("JEV_MODEL")
        response = _get_json(f"{base_url}/v1/models", _env("JEV_API_KEY"))
        models = [item.get("name") for item in response.get("models", [])]
        available = configured_model in models
        return CheckResult(
            service="jev",
            status="ok" if available else "blocked",
            details={"configured_model": configured_model, "available_models": models},
        )
    except Exception as error:
        return _failed("jev", error)


def run_preflight() -> list[CheckResult]:
    configuration = check_configuration()
    if configuration.status != "ok":
        return [configuration]
    return [configuration, check_vultr(), check_huggingface(), check_openrouter(), check_jev()]


def _get_json(url: str, token: str, *, force_ipv4: bool = False) -> dict[str, Any]:
    if any(character in token for character in "\r\n\""):
        raise ValueError("credential contains unsupported characters")
    config = f'header = "Authorization: Bearer {token}"\n'
    command = [
        "curl",
        "--fail-with-body",
        "--silent",
        "--show-error",
        "--max-time",
        "20",
        "--config",
        "-",
    ]
    if force_ipv4:
        command.append("-4")
    command.append(url)
    result = subprocess.run(
        command,
        input=config,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        message = result.stdout or result.stderr or f"curl exit {result.returncode}"
        match = re.search(r'"error"\s*:\s*"([^"]+)"', message)
        if match:
            message = match.group(1)
        status = re.search(r'"status"\s*:\s*(\d+)', result.stdout)
        prefix = f"HTTP {status.group(1)}: " if status else ""
        raise RuntimeError(prefix + _sanitize(message))
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("service returned invalid JSON") from error


def _env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"missing configuration: {name}")
    return value


def _failed(service: str, error: Exception) -> CheckResult:
    return CheckResult(service=service, status="blocked", details={"error": _sanitize(str(error))})


def _sanitize(value: str) -> str:
    value = re.sub(r"Bearer\s+\S+", "Bearer [redacted]", value, flags=re.IGNORECASE)
    value = re.sub(r"(?:hf_|sk-|jv_|tsf_)[A-Za-z0-9_.-]+", "[redacted]", value)
    return " ".join(value.split())[:300]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=Path, default=Path(".env"))
    args = parser.parse_args()
    load_env(args.env)
    results = run_preflight()
    print(json.dumps([asdict(result) for result in results], indent=2))
    if any(result.status != "ok" for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
