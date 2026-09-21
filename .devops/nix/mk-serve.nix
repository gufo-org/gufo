{
  lib,
  gufo ? null,
}:

{
  modality ? "llm", # "llm", "image", "video", "tts", or "asr"
  model ? null, # path, derivation, or string to GGUF model or weights directory

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
  maxTokens ? null,
  temperature ? null,
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
  speculative ? null, # "dspark", "dflash2", "mtp", "off"
  dflashModel ? null,
  dsparkModel ? null,
  mtpModel ? null,
  draftTokens ? null,
  draftPolicy ? null,
  minDraftTokens ? null,
  prefillChunk ? null,
  maxPending ? null,
  maxPendingPerClient ? null,
  requestTimeoutMs ? null,
  maxOutputBytes ? null,
  maxBufferedOutputBytes ? null,
  maxBufferedOutputTotal ? null,

  # Video Options
  root ? null,
  manifest ? null,
  ttl ? null,

  # Custom gufo package override
  gufoPackage ? gufo,
  extraArgs ? [ ],
}:

let
  validModalities = [ "llm" "video" "image" "tts" "asr" ];
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
    "dflash2"
    "mtp"
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
    lib.optionals (model != null) [
      "--model"
      (toString model)
    ]
    ++ lib.optionals (servedModelName != null && lib.elem modality [ "llm" "image" "tts" "asr" ]) [
      "--served-model-name"
      servedModelName
    ]
    ++ lib.optionals (modality == "llm") (
      lib.optionals (context != null) [
        "--context"
        (toString context)
      ]
      ++ lib.optionals (maxTokens != null) [
        "--max-tokens"
        (toString maxTokens)
      ]
      ++ lib.optionals (temperature != null) [
        "--temperature"
        (toString temperature)
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
      ++ lib.optionals (speculative != null) [
        "--speculative"
        speculative
      ]
      ++ lib.optionals (dflashModel != null) [
        "--dflash-model"
        (toString dflashModel)
      ]
      ++ lib.optionals (dsparkModel != null) [
        "--dspark-model"
        (toString dsparkModel)
      ]
      ++ lib.optionals (mtpModel != null) [
        "--mtp-model"
        (toString mtpModel)
      ]
      ++ lib.optionals (draftTokens != null) [
        "--draft-tokens"
        (toString draftTokens)
      ]
      ++ lib.optionals (draftPolicy != null) [
        "--draft-policy"
        draftPolicy
      ]
      ++ lib.optionals (minDraftTokens != null) [
        "--min-draft-tokens"
        (toString minDraftTokens)
      ]
      ++ lib.optionals (prefillChunk != null) [
        "--prefill-chunk"
        (toString prefillChunk)
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
    ++ lib.optionals (lib.elem modality [ "tts" "asr" ] && context != null) [
      "--context"
      (toString context)
    ]
    ++ lib.optionals (modality == "tts") (
      lib.concatMap (
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
assert lib.assertMsg (!lib.elem modality [ "image" "tts" "asr" ] || sessions == null)
  "gufo.mkServe: image and speech requests are queued; 'sessions' is an LLM/video option";
assert lib.assertMsg (model != null && model != "")
  "gufo.mkServe: 'model' must be specified (cannot be empty)";
assert lib.assertMsg (modality == "tts" || voices == { })
  "gufo.mkServe: 'voices' requires modality 'tts'";
assert lib.assertMsg (think == null || lib.elem think validThinkModes)
  "gufo.mkServe: 'think' must be one of ${lib.generators.toJSON { } validThinkModes}, got '${toString think}'";
assert lib.assertMsg (preserveThinking == null || lib.elem preserveThinking validThinkModes)
  "gufo.mkServe: 'preserveThinking' must be one of ${lib.generators.toJSON { } validThinkModes}, got '${toString preserveThinking}'";
assert lib.assertMsg (reasoningEffort == null || lib.elem reasoningEffort validReasoningEfforts)
  "gufo.mkServe: 'reasoningEffort' must be one of ${lib.generators.toJSON { } validReasoningEfforts}, got '${toString reasoningEffort}'";
assert lib.assertMsg (speculative == null || lib.elem speculative validSpeculativeModes)
  "gufo.mkServe: 'speculative' must be one of ${lib.generators.toJSON { } validSpeculativeModes}, got '${toString speculative}'";
"${bin} serve${serverStr}${rawHostStr}${rawPortStr} ${modality}${modalityStr}"
