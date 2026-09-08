import main

# Inspect original_router
for i, r in enumerate(main.app.routes):
    if type(r).__name__ == '_IncludedRouter':
        orig = r.original_router
        print(f"Router {i}: original_router type={type(orig).__name__}")
        for route in orig.routes:
            print(f"  path={getattr(route, 'path', None)} methods={getattr(route, 'methods', None)}")
