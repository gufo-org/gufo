# Automatic API revision derivation

Derive the extension API revision from permanent change records. CMake generates
public version constants from the compatibility series and record count, so
parallel API changes no longer edit a shared version or changelog. Extensions
continue to request a revision floor within their compatibility series.
