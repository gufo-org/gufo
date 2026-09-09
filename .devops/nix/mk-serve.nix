{
  lib,
  gufo ? null,
}:

{
  modality ? "llm", # "llm", "video", "audio" (alias "tts"), or "asr" (alias "stt")
  model ? null, # path, derivation, or string to GGUF model or weights directory

  # Audio Options (modality "audio"/"asr"): one audio server can host
  # Qwen3-TTS synthesis, Qwen3-ASR transcription, or both at once.
  ttsModel ? null,
  asrModel ? null,
  ttsContext ? null,
  asrContext ? null,
  # Named Qwen3-TTS Base voices. Each entry is either a bare reference WAV
  # (its transcript read from a `.txt` sidecar beside it) or an attrset
  # { wav = <path>; text = <transcript or path to a file holding it>; }:
  #
  # An optional `language` supplies the request default for that voice, so a
  # caller naming it need not repeat the language:
  #
  #   voices = {
  #     narrator_eng = "/voices/en.wav";
  #     narrator_ita = {
  #       wav = "/voices/it.wav";
  #       text = "Questo racconto...";
  #       language = "italian";
  #     };
  #     narrator_de = { wav = "/voices/de.wav"; text = "/voices/de.txt"; };
  #   };
  voices ? { },

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
  think ? null, # "on", "off", "auto"
  reasoningEffort ? null, # "auto", "minimal", "low", "medium", "high", "xhigh", "max"
  preserveThinking ? null, # "on", "off", "auto"
  cacheDisk ? null,
  cacheDiskBytes ? null,
  cacheDiskStagingBytes ? null,
  speculative ? null, # "dspark", "dflash", "dflash2", "mtp", "mtp-npu", "npu", "pld", "self", "off"
  specType ? null, # alias for speculative ("draft-mtp" -> "mtp", etc.)
  draftModel ? null, # generic draft model path / derivation
  dflashModel ? null,
  dsparkModel ? null,
  mtpModel ? null,
  draftTokens ? null,
  specDraftNMax ? null, # alias for draftTokens
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
  finalDsparkModel =
    if dsparkModel != null then
      dsparkModel
    else if finalSpeculative == "dspark" then
      draftModel
    else
      null;
  finalMtpModel =
    if mtpModel != null then
      mtpModel
    else if (finalSpeculative != null && finalSpeculative != "dflash" && finalSpeculative != "dflash2" && finalSpeculative != "dspark" && finalSpeculative != "off") then
      draftModel
    else
      null;
  # `gufo serve` has a single "audio" subcommand hosting Qwen3-TTS synthesis,
  # Qwen3-ASR transcription, or both. "tts"/"asr"/"stt" remain accepted helper
  # spellings: they all emit `serve audio`, and "asr"/"stt" additionally route
  # a bare `model` to --asr-model instead of the TTS default.
  isAsrSpelling = modality == "asr" || modality == "stt";
  finalModality =
    if modality == "tts" || isAsrSpelling then "audio" else modality;
  validModalities = [
    "llm"
    "video"
    "audio"
    "tts"
    "asr"
    "stt"
  ];
  validThinkModes = [
    "on"
    "off"
    "auto"
  ];
  validReasoningEfforts = [
    "auto"
    "minimal"
    "low"
    "medium"
    "high"
    "xhigh"
    "max"
  ];
  validSpeculativeModes = [
    "dspark"
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

  isAudio = finalModality == "audio";

  # Under an "asr"/"stt" spelling a bare `model`/`context` names the ASR
  # service; the CLI's own --model/--context are TTS aliases.
  finalTtsModel = if ttsModel != null then ttsModel else null;
  finalAsrModel =
    if asrModel != null then
      asrModel
    else if isAsrSpelling && model != null then
      model
    else
      null;
  finalAsrContext =
    if asrContext != null then
      asrContext
    else if isAsrSpelling then
      finalContext
    else
      null;

  modalityArgs =
    lib.optionals (model != null && !isAsrSpelling) [
      "--model"
      (toString model)
    ]
    ++ lib.optionals (finalModality == "llm") (
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
      ++ lib.optionals (think != null) [
        "--think"
        think
      ]
      ++ lib.optionals (reasoningEffort != null) [
        "--reasoning-effort"
        reasoningEffort
      ]
      ++ lib.optionals (preserveThinking != null) [
        "--preserve-thinking"
        preserveThinking
      ]
      ++ lib.optionals (cacheDisk != null) [
        "--cache-disk"
        (toString cacheDisk)
      ]
      ++ lib.optionals (cacheDiskBytes != null) [
        "--cache-disk-bytes"
        (toString cacheDiskBytes)
      ]
      ++ lib.optionals (cacheDiskStagingBytes != null) [
        "--cache-disk-staging-bytes"
        (toString cacheDiskStagingBytes)
      ]
      ++ lib.optionals (finalSpeculative != null) [
        "--speculative"
        finalSpeculative
      ]
      ++ lib.optionals (finalDflashModel != null) [
        "--dflash-model"
        (toString finalDflashModel)
      ]
      ++ lib.optionals (finalDsparkModel != null) [
        "--dspark-model"
        (toString finalDsparkModel)
      ]
      ++ lib.optionals (finalMtpModel != null && finalDflashModel == null) [
        "--mtp-model"
        (toString finalMtpModel)
      ]
      ++ lib.optionals (finalDraftTokens != null) [
        "--draft-tokens"
        (toString finalDraftTokens)
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
    ++ lib.optionals (finalModality == "video") (
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
    ++ lib.optionals isAudio (
      lib.optionals (finalContext != null && !isAsrSpelling) [
        "--context"
        (toString finalContext)
      ]
      ++ lib.optionals (finalTtsModel != null) [
        "--tts-model"
        (toString finalTtsModel)
      ]
      ++ lib.optionals (finalAsrModel != null) [
        "--asr-model"
        (toString finalAsrModel)
      ]
      ++ lib.optionals (ttsContext != null) [
        "--tts-context"
        (toString ttsContext)
      ]
      ++ lib.optionals (finalAsrContext != null) [
        "--asr-context"
        (toString finalAsrContext)
      ]
      ++ lib.concatMap (
        name:
        let
          entry = voices.${name};
          structured = lib.isAttrs entry && !lib.isDerivation entry;
          wav = if structured then entry.wav else entry;
          text = if structured then (entry.text or null) else null;
          language = if structured then (entry.language or null) else null;
        in
        assert lib.assertMsg (!structured || entry ? wav)
          "gufo.mkServe: voice '${name}' must set 'wav'";
        [
          "--voice"
          "${name}=${toString wav}"
        ]
        ++ lib.optionals (text != null) [
          "--voice-text"
          "${name}=${toString text}"
        ]
        ++ lib.optionals (language != null) [
          "--voice-lang"
          "${name}=${toString language}"
        ]
      ) (builtins.attrNames voices)
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
assert lib.assertMsg (
  (model != null && model != "") || (isAudio && (finalTtsModel != null || finalAsrModel != null))
) "gufo.mkServe: 'model' must be specified (cannot be empty)";
assert lib.assertMsg (isAudio || (ttsModel == null && asrModel == null && ttsContext == null && asrContext == null))
  "gufo.mkServe: 'ttsModel'/'asrModel'/'ttsContext'/'asrContext' require modality 'audio' or 'asr'";
assert lib.assertMsg (isAudio || voices == { })
  "gufo.mkServe: 'voices' requires modality 'audio'";
assert lib.assertMsg (voices == { } || finalTtsModel != null)
  "gufo.mkServe: 'voices' requires a Qwen3-TTS checkpoint ('ttsModel' or 'model')";
assert lib.assertMsg (think == null || lib.elem think validThinkModes)
  "gufo.mkServe: 'think' must be one of ${lib.generators.toJSON { } validThinkModes}, got '${toString think}'";
assert lib.assertMsg (preserveThinking == null || lib.elem preserveThinking validThinkModes)
  "gufo.mkServe: 'preserveThinking' must be one of ${lib.generators.toJSON { } validThinkModes}, got '${toString preserveThinking}'";
assert lib.assertMsg (reasoningEffort == null || lib.elem reasoningEffort validReasoningEfforts)
  "gufo.mkServe: 'reasoningEffort' must be one of ${lib.generators.toJSON { } validReasoningEfforts}, got '${toString reasoningEffort}'";
assert lib.assertMsg (finalSpeculative == null || lib.elem finalSpeculative validSpeculativeModes)
  "gufo.mkServe: 'speculative' must be one of ${lib.generators.toJSON { } validSpeculativeModes}, got '${toString finalSpeculative}'";
"${bin} serve${serverStr}${rawHostStr}${rawPortStr} ${finalModality}${modalityStr}"
