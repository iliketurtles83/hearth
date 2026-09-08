import main

def show_routes(routes, depth=0):
    indent = "  " * depth
    for r in routes:
        name = getattr(r, "name", None)
        path = getattr(r, "path", None)
        methods = getattr(r, "methods", None)
        rtype = type(r).__name__
        
        # Check for _IncludedRouter or similar containers
        for attr in ['routes', 'include', 'router']:
            val = getattr(r, attr, None)
            if val is not None and hasattr(val, '__iter__') and not isinstance(val, str):
                try:
                    print(f"{indent}{rtype}(name={name}, path={path}) -> has '{attr}' ({len(list(val))} items)")
                    # Don't recurse into _IncludedRouter - it's just a wrapper
                    break
                except:
                    pass
        
        if methods is not None:
            print(f"{indent}{rtype}(path={path}, methods={methods})")

show_routes(main.app.routes)
