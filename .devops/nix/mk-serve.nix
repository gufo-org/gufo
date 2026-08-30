{
  lib,
  gufo ? null,
}:

{
  modality ? "llm", # "llm", "video", or "audio"
  model, # path, derivation, or string to GGUF model or weights directory

  # Server Options
  host ? null,
  port ? null,
  sessions ? null,
  maxConnections ? null,
  maxRequestBytes ? null,
  apiKey ? null,
  verbose ? false,

  # LLM Options
  servedModelName ? null,
  context ? null,
  batchSize ? null, # alias for context
  maxTokens ? null,
  temperature ? null,
  temp ? null, # alias for temperature
  topP ? null,
  topK ? null,
  minP ? null,
  minKeep ? null,
  seed ? null,
  repeatPenalty ? null,
  repeatLastN ? null,
  frequencyPenalty ? null,
  presencePenalty ? null,
  system ? null,
  raw ? false,
  chatTemplate ? null,
  think ? null, # "on", "off", "auto"
  reasoningBudget ? null,
  speculative ? null, # "dflash", "dflash2", "mtp", "mtp-npu", "npu", "pld", "self", "off"
  specType ? null, # alias for speculative ("draft-mtp" -> "mtp", etc.)
  draftModel ? null, # generic draft model path / derivation
  dflashModel ? null,
  mtpModel ? null,
  draftTokens ? null,
  specDraftNMax ? null, # alias for draftTokens
  draftPolicy ? null, # "fixed", "rolling", "accepted-ema"
  minDraftTokens ? null,
  specDraftPMin ? null, # alias for speculative confidence / floor
  prefillChunk ? null,
  ubatchSize ? null, # alias for prefillChunk
  maxPending ? null,
  maxPendingPerClient ? null,
  requestTimeoutMs ? null,
  maxOutputBytes ? null,
  maxBufferedOutputBytes ? null,
  maxBufferedOutputTotal ? null,
  cpu ? false,

  # Video Options
  root ? null,
  manifest ? null,
  ttl ? null,

  # Custom gufo package override
  gufoPackage ? gufo,
  extraArgs ? [ ],
}:

let
  finalContext = if context != null then context else batchSize;
  finalTemperature = if temperature != null then temperature else temp;
  finalDraftTokens = if draftTokens != null then draftTokens else specDraftNMax;
  finalPrefillChunk = if prefillChunk != null then prefillChunk else ubatchSize;
  finalSpeculative =
    if specType == "draft-mtp" then
      "mtp"
    else if specType == "draft-dflash" then
      "dflash2"
    else if specType != null then
      specType
    else
      speculative;
  finalDflashModel =
    if dflashModel != null then
      dflashModel
    else if (finalSpeculative == "dflash" || finalSpeculative == "dflash2") then
      draftModel
    else
      null;
  finalMtpModel =
    if mtpModel != null then
      mtpModel
    else if (finalSpeculative != null && finalSpeculative != "dflash" && finalSpeculative != "dflash2" && finalSpeculative != "off") then
      draftModel
    else
      null;
  validModalities = [
    "llm"
    "video"
    "audio"
  ];
  validThinkModes = [
    "on"
    "off"
    "auto"
  ];
  validSpeculativeModes = [
    "dflash"
    "dflash2"
    "mtp"
    "mtp-npu"
    "npu"
    "pld"
    "self"
    "off"
  ];

  isRawEnvVar = val: builtins.isString val && lib.hasPrefix "$" val;

  serverArgs =
    lib.optionals (host != null && !isRawEnvVar host) [
      "--host"
      host
    ]
    ++ lib.optionals (port != null && !isRawEnvVar port) [
      "--port"
      (toString port)
    ]
    ++ lib.optionals (sessions != null) [
      "--sessions"
      (toString sessions)
    ]
    ++ lib.optionals (maxConnections != null) [
      "--max-connections"
      (toString maxConnections)
    ]
    ++ lib.optionals (maxRequestBytes != null) [
      "--max-request-bytes"
      (toString maxRequestBytes)
    ]
    ++ lib.optionals (apiKey != null) [
      "--api-key"
      apiKey
    ]
    ++ lib.optionals verbose [ "--verbose" ];

  modalityArgs =
    [
      "--model"
      (toString model)
    ]
    ++ lib.optionals (modality == "llm") (
      lib.optionals (servedModelName != null) [
        "--served-model-name"
        servedModelName
      ]
      ++ lib.optionals (finalContext != null) [
        "--context"
        (toString finalContext)
      ]
      ++ lib.optionals (maxTokens != null) [
        "--max-tokens"
        (toString maxTokens)
      ]
      ++ lib.optionals (finalTemperature != null) [
        "--temperature"
        (toString finalTemperature)
      ]
      ++ lib.optionals (topP != null) [
        "--top-p"
        (toString topP)
      ]
      ++ lib.optionals (topK != null) [
        "--top-k"
        (toString topK)
      ]
      ++ lib.optionals (minP != null) [
        "--min-p"
        (toString minP)
      ]
      ++ lib.optionals (minKeep != null) [
        "--min-keep"
        (toString minKeep)
      ]
      ++ lib.optionals (seed != null) [
        "--seed"
        (toString seed)
      ]
      ++ lib.optionals (repeatPenalty != null) [
        "--repeat-penalty"
        (toString repeatPenalty)
      ]
      ++ lib.optionals (repeatLastN != null) [
        "--repeat-last-n"
        (toString repeatLastN)
      ]
      ++ lib.optionals (frequencyPenalty != null) [
        "--frequency-penalty"
        (toString frequencyPenalty)
      ]
      ++ lib.optionals (presencePenalty != null) [
        "--presence-penalty"
        (toString presencePenalty)
      ]
      ++ lib.optionals (system != null) [
        "--system"
        system
      ]
      ++ lib.optionals raw [ "--raw" ]
      ++ lib.optionals (chatTemplate != null) [
        "--chat-template"
        chatTemplate
      ]
      ++ lib.optionals (think != null) [
        "--think"
        think
      ]
      ++ lib.optionals (reasoningBudget != null) [
        "--reasoning-budget"
        (toString reasoningBudget)
      ]
      ++ lib.optionals (finalSpeculative != null) [
        "--speculative"
        finalSpeculative
      ]
      ++ lib.optionals (finalDflashModel != null) [
        "--dflash-model"
        (toString finalDflashModel)
      ]
      ++ lib.optionals (finalMtpModel != null && finalDflashModel == null) [
        "--mtp-model"
        (toString finalMtpModel)
      ]
      ++ lib.optionals (finalDraftTokens != null) [
        "--draft-tokens"
        (toString finalDraftTokens)
      ]
      ++ lib.optionals (draftPolicy != null) [
        "--draft-policy"
        draftPolicy
      ]
      ++ lib.optionals (minDraftTokens != null) [
        "--min-draft-tokens"
        (toString minDraftTokens)
      ]
      ++ lib.optionals (specDraftPMin != null) [
        "--spec-draft-p-min"
        (toString specDraftPMin)
      ]
      ++ lib.optionals (finalPrefillChunk != null) [
        "--prefill-chunk"
        (toString finalPrefillChunk)
      ]
      ++ lib.optionals (maxPending != null) [
        "--max-pending"
        (toString maxPending)
      ]
      ++ lib.optionals (maxPendingPerClient != null) [
        "--max-pending-per-client"
        (toString maxPendingPerClient)
      ]
      ++ lib.optionals (requestTimeoutMs != null) [
        "--request-timeout-ms"
        (toString requestTimeoutMs)
      ]
      ++ lib.optionals (maxOutputBytes != null) [
        "--max-output-bytes"
        (toString maxOutputBytes)
      ]
      ++ lib.optionals (maxBufferedOutputBytes != null) [
        "--max-buffered-output-bytes"
        (toString maxBufferedOutputBytes)
      ]
      ++ lib.optionals (maxBufferedOutputTotal != null) [
        "--max-buffered-output-total"
        (toString maxBufferedOutputTotal)
      ]
      ++ lib.optionals cpu [ "--cpu" ]
    )
    ++ lib.optionals (modality == "video") (
      lib.optionals (root != null) [
        "--root"
        (toString root)
      ]
      ++ lib.optionals (manifest != null) [
        "--manifest"
        (toString manifest)
      ]
      ++ lib.optionals (ttl != null) [
        "--ttl"
        (toString ttl)
      ]
    )
    ++ lib.optionals (modality == "audio") (
      lib.optionals (finalContext != null) [
        "--context"
        (toString finalContext)
      ]
    )
    ++ extraArgs;

  bin = if gufoPackage != null then "${gufoPackage}/bin/gufo" else "gufo";
  escapedServerArgs = lib.escapeShellArgs serverArgs;
  escapedModalityArgs = lib.escapeShellArgs modalityArgs;
  rawHostStr = if (host != null && isRawEnvVar host) then " --host ${host}" else "";
  rawPortStr = if (port != null && isRawEnvVar port) then " --port ${port}" else "";
  serverStr = if (escapedServerArgs != "") then " " + escapedServerArgs else "";
  modalityStr = if (escapedModalityArgs != "") then " " + escapedModalityArgs else "";
in
assert lib.assertMsg (lib.elem modality validModalities)
  "gufo.mkServe: 'modality' must be one of ${lib.generators.toJSON { } validModalities}, got '${modality}'";
assert lib.assertMsg (model != null && model != "")
  "gufo.mkServe: 'model' must be specified (cannot be empty)";
assert lib.assertMsg (think == null || lib.elem think validThinkModes)
  "gufo.mkServe: 'think' must be one of ${lib.generators.toJSON { } validThinkModes}, got '${toString think}'";
assert lib.assertMsg (finalSpeculative == null || lib.elem finalSpeculative validSpeculativeModes)
  "gufo.mkServe: 'speculative' must be one of ${lib.generators.toJSON { } validSpeculativeModes}, got '${toString finalSpeculative}'";
"${bin} serve${serverStr}${rawHostStr}${rawPortStr} ${modality}${modalityStr}"
