foo = None

try:
    try:
        state = "bar"
        foo.append(state)

    except Exception as e:
        e.args = ("Appending '"+state+"' failed", *e.args)
        raise

    print(foo[0]) # would raise too

except Exception as e:
    e.message = "foo"
    e.args = ("print(foo) failed: " + str(foo), *e.args)
    raise