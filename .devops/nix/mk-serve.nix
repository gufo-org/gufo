{
  lib,
  gufo ? null,
}:

let
  serve =
    args@{
      modality ? "llm", # host command; availability comes from selected providers
      provider ? null, # metadata validation selector; runtime still probes the artifact
      model ? null, # path, derivation, or string to GGUF model or weights directory

      # Named voices for providers advertising serving.voices. Each entry is either a bare reference WAV
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
      enableYarn ? false, # Flash-Next: required for context above 262144
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
      speculative ? null,
      draftTokens ? null,
      draftPolicy ? null,
      minDraftTokens ? null,
      prefillChunk ? null,
      maxPending ? null,
      maxPendingPerClient ? null,
      requestTimeoutMs ? null,
      stallTimeoutMs ? null,
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
      ...
    }:

    let
      metadata =
        if builtins.isAttrs gufoPackage && gufoPackage ? extensionMetadata then
          gufoPackage.extensionMetadata
        else
          import ./extension-metadata.nix { inherit lib; };
      # Capability-to-command mapping belongs to the host CLI, not individual models.
      commands = {
        text_generation = "llm";
        embeddings = "embedding";
        decision = "decision";
        image_generation = "image";
        video_generation = "video";
        speech_synthesis = "tts";
        speech_recognition = "asr";
      };
      providers = lib.filter (p: builtins.hasAttr p.capability commands) metadata.providers;
      validModalities = lib.unique (map (p: commands.${p.capability}) providers);
      modalityProviders = lib.filter (p: commands.${p.capability} == modality) providers;
      namedProviders = lib.filter (p: provider == null || p.id == provider) modalityProviders;
      camelName =
        name:
        let
          parts = lib.splitString "-" (lib.removePrefix "--" name);
        in
        builtins.head parts
        + lib.concatMapStrings (
          part: lib.toUpper (builtins.substring 0 1 part) + builtins.substring 1 (-1) part
        ) (builtins.tail parts);
      options = lib.filter (
        o:
        lib.any (
          c:
          builtins.elem c [
            "serve"
            "serve ${modality}"
          ]
        ) o.commands
      ) metadata.options;
      optionNames = lib.unique (map (o: camelName o.name) options);
      supplied = builtins.removeAttrs args (builtins.attrNames (builtins.functionArgs serve));
      supports = p: o: !(o ? providers) || o.providers == [ ] || builtins.elem p.id o.providers;
      llmOnlyNames = [
        "enableYarn"
        "maxTokens"
        "temperature"
        "topP"
        "topK"
        "minP"
        "minKeep"
        "seed"
        "repeatPenalty"
        "repeatLastN"
        "frequencyPenalty"
        "presencePenalty"
        "think"
        "reasoningEffort"
        "preserveThinking"
        "cacheDisk"
        "cacheDiskBytes"
        "cacheDiskStagingBytes"
        "speculative"
        "draftTokens"
        "draftPolicy"
        "minDraftTokens"
        "prefillChunk"
        "maxPending"
        "maxPendingPerClient"
        "requestTimeoutMs"
        "stallTimeoutMs"
        "maxOutputBytes"
        "maxBufferedOutputBytes"
        "maxBufferedOutputTotal"
      ];
      active =
        name:
        let
          value = args.${name} or null;
        in
        value != null && value != false;
      activeOptions = lib.filter (o: supplied.${camelName o.name} or null != null) options;
      candidates = lib.filter (
        p:
        (speculative == null || speculative == "off" || builtins.elem speculative (p.speculative or [ ]))
        && lib.all (o: lib.any (other: other.name == o.name && supports p other) options) activeOptions
        && (voices == { } || (p.serving.voices or false))
      ) namedProviders;
      optionTypeValid =
        o: value:
        if value == null then
          true
        else if o.type == "boolean" then
          builtins.isBool value
        else if o.type == "integer" then
          builtins.isInt value
        else if o.type == "number" then
          builtins.isInt value || builtins.isFloat value
        else if o.type == "path" then
          (builtins.isPath value || builtins.isString value || lib.isDerivation value) && toString value != ""
        else
          builtins.isString value;
      generatedArgs = lib.concatMap (
        name:
        let
          o = lib.findFirst (o: camelName o.name == name) null options;
          value = supplied.${name} or null;
        in
        if value == null then
          [ ]
        else if o.type == "boolean" then
          lib.optional value o.name
        else
          [
            o.name
            (toString value)
          ]
      ) optionNames;
      requiredDraftsPresent =
        p:
        let
          flag = p.serving.draft_options.${speculative} or null;
        in
        speculative == null
        || speculative == "off"
        || flag == null
        || (supplied.${camelName flag} or null != null && toString supplied.${camelName flag} != "");
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
        "off"
      ]
      ++ lib.unique (lib.concatMap (p: p.speculative or [ ]) namedProviders);

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
        ++
          lib.optionals
            (
              servedModelName != null
              && lib.elem modality [
                "llm"
                "image"
                "tts"
                "asr"
                "embedding"
                "decision"
              ]
            )
            [
              "--served-model-name"
              servedModelName
            ]
        ++ lib.optionals (modality == "llm") (
          lib.optionals (context != null) [
            "--context"
            (toString context)
          ]
          ++ lib.optionals enableYarn [ "--enable-yarn" ]
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
          ++ lib.optionals (stallTimeoutMs != null) [
            "--stall-timeout-ms"
            (toString stallTimeoutMs)
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
        ++
          lib.optionals
            (
              lib.elem modality [
                "tts"
                "asr"
                "decision"
              ]
              && context != null
            )
            [
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
            assert lib.assertMsg (!structured || entry ? wav) "gufo.mkServe: voice '${name}' must set 'wav'";
            assert lib.assertMsg (
              (builtins.isString wav || builtins.isPath wav || lib.isDerivation wav) && toString wav != ""
            ) "gufo.mkServe: voice '${name}' requires a nonempty WAV path";
            assert lib.assertMsg (
              !structured
              ||
                builtins.removeAttrs entry [
                  "wav"
                  "text"
                  "language"
                ] == { }
            ) "gufo.mkServe: voice '${name}' has unknown fields";
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
        ++ generatedArgs
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
      "gufo.mkServe: 'modality' must be one of ${
        lib.generators.toJSON { } validModalities
      }, got '${modality}'";
    assert lib.assertMsg (
      !lib.elem modality [
        "image"
        "tts"
        "asr"
        "embedding"
        "decision"
      ]
      || sessions == null
    ) "gufo.mkServe: image and speech requests are queued; 'sessions' is an LLM/video option";
    assert lib.assertMsg (
      model != null && model != ""
    ) "gufo.mkServe: 'model' must be specified (cannot be empty)";
    assert lib.assertMsg (
      modality == "llm" || !lib.any active llmOnlyNames
    ) "gufo.mkServe: LLM options are unsupported for modality '${modality}'";
    assert lib.assertMsg (
      modality == "video"
      || !lib.any active [
        "root"
        "manifest"
        "ttl"
      ]
    ) "gufo.mkServe: video options are unsupported for modality '${modality}'";
    assert lib.assertMsg (
      context == null
      || lib.elem modality [
        "llm"
        "tts"
        "asr"
        "decision"
      ]
    ) "gufo.mkServe: context is unsupported for modality '${modality}'";
    assert lib.assertMsg (builtins.attrNames (builtins.removeAttrs supplied optionNames) == [ ])
      "gufo.mkServe: unknown arguments: ${lib.concatStringsSep ", " (builtins.attrNames (builtins.removeAttrs supplied optionNames))}";
    assert lib.assertMsg (lib.all (
      o:
      lib.all (
        other: camelName o.name != camelName other.name || (o.name == other.name && o.type == other.type)
      ) options
    ) options) "gufo.mkServe: conflicting extension option names or types";
    assert lib.assertMsg (lib.all (
      o: !(builtins.hasAttr (camelName o.name) (builtins.functionArgs serve))
    ) options) "gufo.mkServe: extension option shadows a host argument";
    assert lib.assertMsg (namedProviders != [ ])
      "gufo.mkServe: provider '${toString provider}' is unavailable for modality '${modality}' (${metadata.selectionMode})";
    assert lib.assertMsg (candidates != [ ])
      "gufo.mkServe: unsupported options/speculation/voices for providers ${
        lib.concatMapStringsSep ", " (p: p.id) namedProviders
      }";
    assert lib.assertMsg (lib.all (
      o: optionTypeValid o (supplied.${camelName o.name} or null)
    ) options) "gufo.mkServe: extension option has an invalid type or empty path";
    assert lib.assertMsg (lib.any requiredDraftsPresent candidates)
      "gufo.mkServe: speculative '${toString speculative}' requires a draft model path for ${
        lib.concatMapStringsSep ", " (p: p.id) candidates
      }";
    assert lib.assertMsg (think == null || lib.elem think validThinkModes)
      "gufo.mkServe: 'think' must be one of ${
        lib.generators.toJSON { } validThinkModes
      }, got '${toString think}'";
    assert lib.assertMsg (preserveThinking == null || lib.elem preserveThinking validThinkModes)
      "gufo.mkServe: 'preserveThinking' must be one of ${
        lib.generators.toJSON { } validThinkModes
      }, got '${toString preserveThinking}'";
    assert lib.assertMsg (reasoningEffort == null || lib.elem reasoningEffort validReasoningEfforts)
      "gufo.mkServe: 'reasoningEffort' must be one of ${
        lib.generators.toJSON { } validReasoningEfforts
      }, got '${toString reasoningEffort}'";
    assert lib.assertMsg (speculative == null || lib.elem speculative validSpeculativeModes)
      "gufo.mkServe: 'speculative' must be one of ${
        lib.generators.toJSON { } validSpeculativeModes
      }, got '${toString speculative}'";
    "${bin} serve${serverStr}${rawHostStr}${rawPortStr} ${modality}${modalityStr}"

  ;
in
serve
