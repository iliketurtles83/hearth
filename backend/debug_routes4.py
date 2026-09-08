import main

# Inspect _IncludedRouter attributes
for i, r in enumerate(main.app.routes):
    if type(r).__name__ == '_IncludedRouter':
        print(f"Router {i}: dir={dir(r)}")
        # Check for private attrs that hold the inner router
        for attr in dir(r):
            if not attr.startswith('__'):
                val = getattr(r, attr)
                if hasattr(val, '__iter__') and not isinstance(val, str):
                    try:
                        items = list(val)
                        print(f"  {attr}: {len(items)} items")
                        if items:
                            print(f"    first: type={type(items[0]).__name__} path={getattr(items[0], 'path', None)} methods={getattr(items[0], 'methods', None)}")
                    except:
                        pass
