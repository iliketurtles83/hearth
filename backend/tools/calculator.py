import ast
import math
import operator
import re
import logging
from typing import Dict, Any

from tools.base import ToolResult
logger = logging.getLogger("assistant.tools.calculator")

# Mathematical functions and constants allowed
ALLOWED_MATH = {
    'sqrt': math.sqrt,
    'abs': abs,
    'round': round,
    'sin': math.sin,
    'cos': math.cos,
    'tan': math.tan,
    'log': math.log,
    'log10': math.log10,
    'ceil': math.ceil,
    'floor': math.floor,
    'pi': math.pi,
    'e': math.e,
}

# Supported operators
ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

def safe_eval(node):
    if isinstance(node, ast.Expression):
        return safe_eval(node.body)
    elif isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value)}")
    elif isinstance(node, ast.BinOp):
        left = safe_eval(node.left)
        right = safe_eval(node.right)
        op = ALLOWED_OPERATORS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")
        return op(left, right)
    elif isinstance(node, ast.UnaryOp):
        operand = safe_eval(node.operand)
        op = ALLOWED_OPERATORS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op(operand)
    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls are allowed")
        func_name = node.func.id
        if func_name not in ALLOWED_MATH:
            raise ValueError(f"Unsupported function: {func_name}")
        func = ALLOWED_MATH[func_name]
        args = [safe_eval(arg) for arg in node.args]
        return func(*args)
    elif isinstance(node, ast.Name):
        if node.id in ALLOWED_MATH:
            return ALLOWED_MATH[node.id]
        raise ValueError(f"Unsupported variable: {node.id}")
    else:
        raise ValueError(f"Unsupported expression node: {type(node).__name__}")

def preprocess_percentages(expr: str) -> str:
    # "15% of 87.50" -> "(0.15 * 87.50)"
    expr = re.sub(r'([\d\.]+)%\s*of\s*([\d\.]+)', lambda m: f"({float(m.group(1)) / 100} * {m.group(2)})", expr, flags=re.IGNORECASE)
    
    # "87.50 + 15%" -> "(87.50 * 1.15)"
    expr = re.sub(r'([\d\.]+)\s*\+\s*([\d\.]+)%', lambda m: f"({m.group(1)} * {1 + float(m.group(2)) / 100})", expr)
    
    # "100 - 20%" -> "(100 * 0.80)"
    expr = re.sub(r'([\d\.]+)\s*-\s*([\d\.]+)%', lambda m: f"({m.group(1)} * {1 - float(m.group(2)) / 100})", expr)
    
    return expr

CONVERSIONS = {
    # Length
    ('miles', 'km'): lambda x: x * 1.60934,
    ('km', 'miles'): lambda x: x / 1.60934,
    ('miles', 'kilometers'): lambda x: x * 1.60934,
    ('kilometers', 'miles'): lambda x: x / 1.60934,
    ('feet', 'meters'): lambda x: x * 0.3048,
    ('meters', 'feet'): lambda x: x / 0.3048,
    ('inches', 'cm'): lambda x: x * 2.54,
    ('cm', 'inches'): lambda x: x / 2.54,
    ('yards', 'meters'): lambda x: x * 0.9144,
    ('meters', 'yards'): lambda x: x / 0.9144,
    
    # Weight
    ('pounds', 'kg'): lambda x: x * 0.453592,
    ('kg', 'pounds'): lambda x: x / 0.453592,
    ('ounces', 'grams'): lambda x: x * 28.3495,
    ('grams', 'ounces'): lambda x: x / 28.3495,
    
    # Temperature
    ('celsius', 'fahrenheit'): lambda x: (x * 9/5) + 32,
    ('fahrenheit', 'celsius'): lambda x: (x - 32) * 5/9,
    ('celsius', 'kelvin'): lambda x: x + 273.15,
    ('kelvin', 'celsius'): lambda x: x - 273.15,
    
    # Volume
    ('liters', 'gallons'): lambda x: x * 0.264172,
    ('gallons', 'liters'): lambda x: x / 0.264172,
    ('cups', 'ml'): lambda x: x * 236.588,
    ('ml', 'cups'): lambda x: x / 236.588,
    
    # Speed
    ('mph', 'kph'): lambda x: x * 1.60934,
    ('kph', 'mph'): lambda x: x / 1.60934,
    
    # Data
    ('bytes', 'kb'): lambda x: x / 1024,
    ('kb', 'bytes'): lambda x: x * 1024,
    ('kb', 'mb'): lambda x: x / 1024,
    ('mb', 'kb'): lambda x: x * 1024,
    ('mb', 'gb'): lambda x: x / 1024,
    ('gb', 'mb'): lambda x: x * 1024,
    ('gb', 'tb'): lambda x: x / 1024,
    ('tb', 'gb'): lambda x: x * 1024,
}

def parse_conversion(prompt: str):
    patterns = [
        r'convert\s+([\d\.]+)\s+([a-zA-Z]+)\s+to\s+([a-zA-Z]+)',
        r'([\d\.]+)\s+([a-zA-Z]+)\s+in\s+([a-zA-Z]+)',
        r'([\d\.]+)\s+([a-zA-Z]+)\s+to\s+([a-zA-Z]+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, prompt, re.IGNORECASE)
        if match:
            val = float(match.group(1))
            from_u = match.group(2).lower()
            to_u = match.group(3).lower()
            return val, from_u, to_u
    return None

async def run(params: Dict[str, Any]) -> ToolResult:
    prompt = params.get("prompt", "").strip()
    expr = params.get("expression", prompt).strip()

    conv_match = parse_conversion(prompt)
    if conv_match:
        val, from_unit, to_unit = conv_match
        conv_func = CONVERSIONS.get((from_unit, to_unit))
        if not conv_func:
            return ToolResult(ok=False, error=f"Conversion from {from_unit} to {to_unit} is not supported.")
        try:
            res = conv_func(val)
            formatted = f"{val} {from_unit} = {res:.2f} {to_unit}"
            return ToolResult(ok=True, data={
                "type": "conversion",
                "value": val,
                "from_unit": from_unit,
                "to_unit": to_unit,
                "result": res,
                "formatted": formatted
            })
        except Exception as e:
            return ToolResult(ok=False, error=str(e))
    
    if not expr:
        return ToolResult(ok=False, error="No expression provided.")

    try:
        processed_expr = preprocess_percentages(expr)
        tree = ast.parse(processed_expr, mode='eval')
        result = safe_eval(tree)
        return ToolResult(ok=True, data={
            "type": "calculation",
            "expression": processed_expr,
            "result": str(result),
            "original": expr
        })
    except ZeroDivisionError:
        return ToolResult(ok=False, error="Division by zero")
    except Exception as e:
        logger.error(f"Error evaluating expression: {e}")
        return ToolResult(ok=False, error=f"Error evaluating expression: {e}")

def format_calculator_response(result: ToolResult) -> str:
    if result.error:
        return f"Error: {result.error}"
    data = result.data
    if data.get("type") == "conversion":
        return data["formatted"]
    elif data.get("type") == "calculation":
        res_val = float(data["result"])
        formatted_res = f"{res_val:.2f}" if res_val % 1 != 0 else str(int(res_val))
        return f"{data['original']} = **{formatted_res}**"
    return "Unknown result format."

import sys as _sys
from tools import register as _register
_register("calculator", _sys.modules[__name__])

