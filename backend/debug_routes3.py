import main

# Inspect the _IncludedRouter contents
for i, r in enumerate(main.app.routes):
    if type(r).__name__ == '_IncludedRouter':
        print(f"Router {i}:")
        inner = list(r.routes)
        print(f"  -> {len(inner)} inner routes")
        for j, ir in enumerate(inner):
            print(f"    {j}: type={type(ir).__name__} path={getattr(ir, 'path', None)} methods={getattr(ir, 'methods', None)}")
