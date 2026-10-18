from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping


MONEY = re.compile(r"[^0-9.\-]")


def validate_output(output: Any) -> dict[str, Any]:
    """Perform content-independent checks without teaching the model new answers."""
    result: dict[str, Any] = {
        "structured": isinstance(output, Mapping),
        "arithmetic_checked": False,
        "arithmetic_valid": None,
    }
    if not isinstance(output, Mapping):
        return result

    subtotal = _decimal(output.get("subtotal"))
    tax = _decimal(output.get("tax"))
    total = _decimal(output.get("total"))
    if subtotal is not None and tax is not None and total is not None:
        result["arithmetic_checked"] = True
        result["arithmetic_valid"] = abs(subtotal + tax - total) <= Decimal("0.02")
    return result


def _decimal(value: Any) -> Decimal | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return Decimal(MONEY.sub("", str(value)))
    except InvalidOperation:
        return None
