import pytest
import sys

class _FakeMemory:
    pass

# Stub out chromadb / memory loading
sys.modules["memory"] = _FakeMemory()

from tools.calculator import run, format_calculator_response
from tools.base import ToolResult

@pytest.mark.asyncio
async def test_basic_arithmetic():
    res = await run({"prompt": "2 + 3"})
    assert res.data["result"] == "5"

    res = await run({"prompt": "10 * 5"})
    assert res.data["result"] == "50"

    res = await run({"prompt": "100 / 4"})
    assert res.data["result"] == "25.0"

@pytest.mark.asyncio
async def test_percentage_of():
    res = await run({"prompt": "15% of 87.50"})
    assert float(res.data["result"]) == 13.125

@pytest.mark.asyncio
async def test_percentage_tip():
    res = await run({"prompt": "87.50 + 15%"})
    assert abs(float(res.data["result"]) - 100.625) < 0.001

@pytest.mark.asyncio
async def test_percentage_discount():
    res = await run({"prompt": "100 - 20%"})
    assert float(res.data["result"]) == 80.0

@pytest.mark.asyncio
async def test_exponentiation():
    res = await run({"prompt": "2 ** 16"})
    assert res.data["result"] == "65536"

@pytest.mark.asyncio
async def test_math_functions():
    res = await run({"prompt": "sqrt(144)"})
    assert float(res.data["result"]) == 12.0

@pytest.mark.asyncio
async def test_nested_expressions():
    res = await run({"prompt": "(2 + 3) * (4 - 1)"})
    assert res.data["result"] == "15"

@pytest.mark.asyncio
async def test_division_by_zero():
    res = await run({"prompt": "10 / 0"})
    assert res.error == "Division by zero"

@pytest.mark.asyncio
async def test_dangerous_input_rejected():
    res = await run({"prompt": "__import__('os').system('ls')"})
    assert res.error is not None
    assert "Unsupported" in res.error or "Error" in res.error

@pytest.mark.asyncio
async def test_attribute_access_rejected():
    res = await run({"prompt": "'hello'.upper()"})
    assert res.error is not None

@pytest.mark.asyncio
async def test_unit_conversion_miles_to_km():
    res = await run({"prompt": "5 miles to km"})
    assert res.data["type"] == "conversion"
    assert abs(res.data["result"] - 8.0467) < 0.01

@pytest.mark.asyncio
async def test_unit_conversion_celsius_to_fahrenheit():
    res = await run({"prompt": "100 celsius to fahrenheit"})
    assert res.data["result"] == 212

@pytest.mark.asyncio
async def test_unit_conversion_pounds_to_kg():
    res = await run({"prompt": "150 pounds to kg"})
    assert abs(res.data["result"] - 68.0388) < 0.01

@pytest.mark.asyncio
async def test_format_calculation():
    res = ToolResult(ok=True, data={"type": "calculation", "result": "13.125", "original": "15% of 87.50", "expression": "(0.15 * 87.50)"})
    formatted = format_calculator_response(res)
    assert "15% of 87.50 = **13.12**" in formatted or "13.13" in formatted

@pytest.mark.asyncio
async def test_format_conversion():
    res = ToolResult(ok=True, data={"type": "conversion", "formatted": "5.0 miles = 8.05 km"})
    formatted = format_calculator_response(res)
    assert formatted == "5.0 miles = 8.05 km"

@pytest.mark.asyncio
async def test_unknown_unit():
    res = await run({"prompt": "10 unknown to other"})
    assert res.error is not None
    assert "not supported" in res.error
