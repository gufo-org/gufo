# Pure evaluation coverage: no GPU package needs to be realized.
{ lib }:
let
  metadata = import ./extension-metadata.nix { inherit lib; };
  mk = import ./mk-serve.nix {
    inherit lib;
    gufo = "/gufo-test";
  };
  render = args: mk ({ model = "/models/main"; } // args);
  fails = args: !(builtins.tryEval (builtins.deepSeq (render args) true)).success;
  contains = text: args: lib.hasInfix text (render args);
  externalMetadata = import ./extension-metadata.nix {
    inherit lib;
    extensions = [ ../../examples/extension_template ];
  };
  externalParentMetadata = import ./extension-metadata.nix {
    inherit lib;
    extensions = [ ../../examples ];
  };
  lockedMetadata = import ./extension-metadata.nix {
    inherit lib;
    extensions = [ ../../examples/extension_template ];
    extensionsFromLock = true;
  };
  # Exercise every declared option type and shared names across providers.
  fixtureOptions =
    map
      (
        o:
        o
        // {
          commands = [ "serve" ];
          providers = [ "fixture" ];
        }
      )
      [
        {
          name = "--test-boolean";
          type = "boolean";
        }
        {
          name = "--test-integer";
          type = "integer";
        }
        {
          name = "--test-number";
          type = "number";
        }
        {
          name = "--test-string";
          type = "string";
        }
        {
          name = "--test-path";
          type = "path";
        }
      ];
  fixtureMetadata = metadata // {
    providers = metadata.providers ++ [
      {
        id = "fixture";
        capability = "text_generation";
        speculative = [ "fixture-draft" ];
        serving.draft_options.fixture-draft = "--test-path";
      }
    ];
    options =
      metadata.options
      ++ fixtureOptions
      ++ [
        {
          name = "--dflash-model";
          type = "path";
          commands = [ "serve" ];
          providers = [ "fixture" ];
        }
      ];
  };
  fixturePackage = {
    outPath = "/fixture";
    extensionMetadata = fixtureMetadata;
  };
  tests = {
    default = render { } == "/gufo-test/bin/gufo serve llm --model /models/main";
    qwen = contains "--dflash-model /draft" {
      provider = "qwen27-q4";
      speculative = "dflash2";
      dflashModel = "/draft";
    };
    flash = contains "--mtp-model /draft" {
      provider = "flash-next-q4";
      speculative = "mtp";
      mtpModel = "/draft";
    };
    deepseek = contains "--dspark-model /draft" {
      provider = "deepseek4";
      speculative = "dspark";
      dsparkModel = "/draft";
    };
    providerValidationOnly = !(contains "--provider" { provider = "deepseek4"; });
    missingDraft = fails { speculative = "mtp"; };
    emptyDraft = fails {
      speculative = "dflash2";
      dflashModel = "";
    };
    wrongDraft = fails {
      provider = "deepseek4";
      dflashModel = "/draft";
    };
    wrongMode = fails {
      provider = "qwen27-q4";
      speculative = "mtp";
      mtpModel = "/draft";
    };
    mixedDrafts = fails {
      dsparkModel = "/d";
      mtpModel = "/m";
    };
    unknownArgument = fails { draftModel = "/draft"; };
    unknownProvider = fails { provider = "absent"; };
    unknownModality = fails { modality = "absent"; };
    missingModel = fails { model = null; };
    tts = contains "--voice-lang 'narrator=italian'" {
      modality = "tts";
      voices.narrator = {
        wav = "/voice.wav";
        text = "hello";
        language = "italian";
      };
    };
    badVoice = fails {
      modality = "tts";
      voices.narrator.text = "missing WAV";
    };
    emptyVoice = fails {
      modality = "tts";
      voices.narrator = "";
    };
    wrongVoiceField = fails {
      modality = "tts";
      voices.narrator = {
        wav = "/voice.wav";
        typo = true;
      };
    };
    speechSpeculation = fails {
      modality = "tts";
      speculative = "off";
    };
    imageContext = fails {
      modality = "image";
      context = 1024;
    };
    wrongVideoOption = fails { manifest = "/manifest.json"; };
    wrongVoices = fails { voices.narrator = "/voice.wav"; };
    speechSessions = fails {
      modality = "asr";
      sessions = 2;
    };
    image = contains " image --model" { modality = "image"; };
    video = contains "--manifest /manifest.json" {
      modality = "video";
      manifest = "/manifest.json";
    };
    asr = contains "asr --model /models/main --context 1024" {
      modality = "asr";
      context = 1024;
    };
    embedding = contains " embedding --model" { modality = "embedding"; };
    external = contains " embedding --model" {
      modality = "embedding";
      provider = "template-model";
      gufoPackage = {
        outPath = "/external";
        extensionMetadata = externalMetadata;
      };
    };
    lockExcludesExternal = fails {
      modality = "embedding";
      provider = "template-model";
      gufoPackage = {
        outPath = "/release";
        extensionMetadata = lockedMetadata;
      };
    };
    externalParent = externalParentMetadata.extensions == externalMetadata.extensions;
    conflictingTypes = fails {
      gufoPackage = fixturePackage // {
        extensionMetadata = fixtureMetadata // {
          options = fixtureMetadata.options ++ [
            {
              name = "--test-integer";
              type = "path";
              commands = [ "serve" ];
              providers = [ "fixture" ];
            }
          ];
        };
      };
    };
    hostShadow = fails {
      gufoPackage = fixturePackage // {
        extensionMetadata = fixtureMetadata // {
          options = fixtureMetadata.options ++ [
            {
              name = "--extra-args";
              type = "string";
              commands = [ "serve" ];
              providers = [ "fixture" ];
            }
          ];
        };
      };
    };
    camelCollision = fails {
      gufoPackage = fixturePackage // {
        extensionMetadata = fixtureMetadata // {
          options = fixtureMetadata.options ++ [
            {
              name = "--test--integer";
              type = "integer";
              commands = [ "serve" ];
              providers = [ "fixture" ];
            }
          ];
        };
      };
    };
    packageSelection =
      lockedMetadata.selectionMode == "lock" && externalMetadata.selectionMode == "discovery";
    generatedTypes =
      contains
        "--test-boolean --test-integer 3 --test-number 0.500000 --test-string 'a b' --test-path /draft"
        {
          gufoPackage = fixturePackage;
          provider = "fixture";
          testBoolean = true;
          testInteger = 3;
          testNumber = 0.5;
          testString = "a b";
          testPath = "/draft";
        };
    falseBoolean =
      !(contains "--test-boolean" {
        gufoPackage = fixturePackage;
        testBoolean = false;
      });
    invalidType = fails {
      gufoPackage = fixturePackage;
      testInteger = "3";
    };
    externalDraft = contains "--speculative fixture-draft" {
      gufoPackage = fixturePackage;
      speculative = "fixture-draft";
      testPath = "/draft";
    };
    externalMissingDraft = fails {
      gufoPackage = fixturePackage;
      speculative = "fixture-draft";
    };
    sharedOption = contains "--dflash-model /draft" {
      gufoPackage = fixturePackage;
      provider = "fixture";
      dflashModel = "/draft";
    };
    shellQuoting = contains "--model '/models/a b'" { model = "/models/a b"; };
  };
  failed = builtins.attrNames (lib.filterAttrs (_: pass: !pass) tests);
in
assert lib.assertMsg (failed == [ ]) "mkServe tests failed: ${lib.concatStringsSep ", " failed}";
{
  passed = builtins.attrNames tests;
}
