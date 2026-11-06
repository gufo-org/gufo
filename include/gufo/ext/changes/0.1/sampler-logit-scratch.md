# Sampler logit scratch storage

Sampling (`sampling.hpp`): `SamplerState` keeps a private `LogitRowScratch`
for the block-maxima top-p/top-k sampler. Source compatible; extensions that
embed `SamplerState` rebuild.
