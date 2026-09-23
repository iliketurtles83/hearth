import asyncio
import json
import time
import httpx

BASE_URL = "http://localhost:10000/v1"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather or forecast for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name, e.g. Helsinki, Tallinn, Paris"},
                    "timeframe": {"type": "string", "description": "today, tomorrow, next week, etc."}
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "music_control",
            "description": "Control music playback or search and play songs/artists/genres",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["play", "pause", "resume", "next", "stop", "shuffle", "now_playing", "queue"],
                        "description": "Playback action"
                    },
                    "query": {"type": "string", "description": "Track, artist, album, or genre name to search and play"},
                    "artist": {"type": "string", "description": "Artist name if specifically requesting artist tracks"}
                },
                "required": ["action"]
            }
        }
    }
]

TEST_CASES = [
    # 1. Simple direct weather
    {
        "name": "weather_direct",
        "messages": [{"role": "user", "content": "What is the weather in Helsinki today?"}],
        "expected_tool": "get_weather",
        "expected_args": {"location": "Helsinki"}
    },
    # 2. Weather follow-up (the core problem discussed earlier!)
    {
        "name": "weather_followup",
        "messages": [
            {"role": "user", "content": "What's the weather like in Paris?"},
            {"role": "assistant", "content": "It's currently 18°C and partly cloudy in Paris."},
            {"role": "user", "content": "What about tomorrow?"}
        ],
        "expected_tool": "get_weather",
        "expected_args": {"location": "Paris"}
    },
    # 3. Weather follow-up location switch
    {
        "name": "weather_location_switch",
        "messages": [
            {"role": "user", "content": "What's the weather like in Paris?"},
            {"role": "assistant", "content": "It's currently 18°C and partly cloudy in Paris."},
            {"role": "user", "content": "And in Tokyo?"}
        ],
        "expected_tool": "get_weather",
        "expected_args": {"location": "Tokyo"}
    },
    # 4. Music direct play
    {
        "name": "music_play_artist",
        "messages": [{"role": "user", "content": "Play some Radiohead"}],
        "expected_tool": "music_control",
        "expected_args": {"action": "play"}
    },
    # 5. Music natural language phrasing (which fastpath regex struggles with)
    {
        "name": "music_natural_phrase",
        "messages": [{"role": "user", "content": "I'm in the mood for something chill like jazz."}],
        "expected_tool": "music_control",
        "expected_args": {"action": "play"}
    },
    # 6. Music playback control
    {
        "name": "music_pause",
        "messages": [{"role": "user", "content": "Can you pause the music please?"}],
        "expected_tool": "music_control",
        "expected_args": {"action": "pause"}
    },
    # 7. Conversational greeting (should NOT trigger tool)
    {
        "name": "social_greeting",
        "messages": [{"role": "user", "content": "Good morning! Hope you're having a good day."}],
        "expected_tool": None
    },
    # 8. General knowledge question (should NOT trigger tool)
    {
        "name": "general_knowledge",
        "messages": [{"role": "user", "content": "Why is the sky blue?"}],
        "expected_tool": None
    },
    # 9. Follow-up conversational clarification (should NOT trigger tool)
    {
        "name": "conversation_followup",
        "messages": [
            {"role": "user", "content": "Tell me a fun fact about octopuses."},
            {"role": "assistant", "content": "Octopuses have three hearts and blue blood!"},
            {"role": "user", "content": "Why do they have three hearts?"}
        ],
        "expected_tool": None
    },
    # 10. Ambiguous / multi-concept request
    {
        "name": "weather_metaphor",
        "messages": [{"role": "user", "content": "I'm feeling a bit under the weather today."}],
        "expected_tool": None
    }
]

async def run_test(client: httpx.AsyncClient, tc: dict):
    payload = {
        "model": "gemma-4",
        "messages": tc["messages"],
        "tools": TOOLS,
        "tool_choice": "auto",
        "temperature": 0.1
    }
    t0 = time.time()
    resp = await client.post(f"{BASE_URL}/chat/completions", json=payload, timeout=30.0)
    dur = time.time() - t0
    data = resp.json()
    
    choice = data.get("choices", [{}])[0]
    msg = choice.get("message", {})
    tool_calls = msg.get("tool_calls", [])
    content = msg.get("content", "")
    
    passed = False
    details = {}
    
    if tc["expected_tool"] is None:
        if not tool_calls:
            passed = True
            details = {"response_preview": content[:60]}
        else:
            passed = False
            details = {"hallucinated_tool": [t["function"]["name"] for t in tool_calls]}
    else:
        if tool_calls:
            first_tool = tool_calls[0]["function"]
            tool_name = first_tool["name"]
            try:
                args = json.loads(first_tool.get("arguments", "{}"))
            except Exception:
                args = {"raw": first_tool.get("arguments")}
            
            tool_match = (tool_name == tc["expected_tool"])
            # check expected args subset
            args_match = True
            for k, v in tc.get("expected_args", {}).items():
                val = str(args.get(k, "")).lower()
                if str(v).lower() not in val:
                    args_match = False
            
            passed = tool_match and args_match
            details = {
                "called_tool": tool_name,
                "args": args,
                "tool_match": tool_match,
                "args_match": args_match
            }
        else:
            passed = False
            details = {"no_tool_called": True, "response_preview": content[:60]}
            
    return {
        "name": tc["name"],
        "passed": passed,
        "duration_sec": round(dur, 2),
        "details": details
    }

async def main():
    print("Starting Gemma 4 Native Tool-Calling Benchmark on local llama-server...\n")
    async with httpx.AsyncClient() as client:
        results = []
        for i, tc in enumerate(TEST_CASES, 1):
            print(f"[{i}/{len(TEST_CASES)}] Testing {tc['name']}... ", end="", flush=True)
            res = await run_test(client, tc)
            results.append(res)
            status = "PASS" if res["passed"] else "FAIL"
            print(f"{status} ({res['duration_sec']}s)")
            if not res["passed"]:
                print(f"     Details: {res['details']}")
        
        passed_count = sum(1 for r in results if r["passed"])
        avg_dur = sum(r["duration_sec"] for r in results) / len(results)
        print("\n" + "="*50)
        print(f"Summary: {passed_count}/{len(results)} passed ({passed_count/len(results)*100:.1f}%)")
        print(f"Average latency per decision: {avg_dur:.2f}s")
        print("="*50)
        
        with open("/home/jack/Projects/hearth/backend/scratch/tool_calling_benchmark.json", "w") as f:
            json.dump({"results": results, "summary": {"passed": passed_count, "total": len(results), "avg_dur": avg_dur}}, f, indent=2)

if __name__ == "__main__":
    asyncio.run(main())
