import main

# Check what's actually in app.routes
print(f"Total routes: {len(main.app.routes)}")
for i, r in enumerate(main.app.routes):
    print(f"  {i}: type={type(r).__name__} path={getattr(r, 'path', None)} methods={getattr(r, 'methods', None)}")
    if hasattr(r, 'routes'):
        inner = list(r.routes)
        print(f"       -> inner routes: {len(inner)}")
        for j, ir in enumerate(inner):
            print(f"         {j}: type={type(ir).__name__} path={getattr(ir, 'path', None)} methods={getattr(ir, 'methods', None)}")

# Check the router functions
print("\n--- Router functions ---")
print(f"create_auth_router: {main.create_auth_router}")
print(f"create_chat_router: {main.create_chat_router}")
print(f"create_voice_router: {main.create_voice_router}")
print(f"create_tts_router: {main.create_tts_router}")

# Check if the routers were called
print("\n--- Services check ---")
print(f"services: {main.services}")
if main.services:
    print(f"  tts: {main.services.tts}")
    print(f"  get_whisper_model: {main.services.get_whisper_model}")
