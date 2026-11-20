"""Transport coverage shared by the suites: every option, not every combination.

A streamed reply re-parses partial output, then must end in the same final
parse a buffered reply returns, so parse checks use the streamed mode. Cases
that vary an option (endpoint, stream, tool choice) take one each, rotating so
consecutive cases cover every option.
"""


def streaming_modes():
    return (True,)


def rotate(options, index):
    options = tuple(options)
    return (options[index % len(options)],)
