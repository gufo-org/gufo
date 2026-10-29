# Changelog

Notable user-facing changes are recorded here. Gufo follows
[Semantic Versioning](https://semver.org/) under the compatibility policy in
[the release guide](docs/RELEASING.md).

## [0.27.0](https://github.com/gufo-org/gufo/compare/v0.26.0...v0.27.0) (2026-10-29)


### Features

* **minimax_h3:** run MiniMax H3 on Windows (MP4 mux, prompt-encoder loading, rocBLAS fallbacks) ([#856](https://github.com/gufo-org/gufo/issues/856)) ([6feba90](https://github.com/gufo-org/gufo/commit/6feba9060477e0d5358deee40ca978c37f3b3d77))
* **serve:** add cache_persist=false to keep a request's prompt off the disk cache ([#811](https://github.com/gufo-org/gufo/issues/811)) ([b7a2d61](https://github.com/gufo-org/gufo/commit/b7a2d616f09cb64b7c438ad86762e02aa8d10c51))
* **serve:** support generated token logprobs ([#814](https://github.com/gufo-org/gufo/issues/814)) ([e590d17](https://github.com/gufo-org/gufo/commit/e590d170f903967d7547c1d0d7037e1ce3c03ab4))


### Bug Fixes

* **ci:** pack the Windows release ZIP without ./ entry names ([#868](https://github.com/gufo-org/gufo/issues/868)) ([e011849](https://github.com/gufo-org/gufo/commit/e011849cab12f78b3284c3eb6a917681be1abf83))
* **loader:** retry any failed unbuffered weight read through the buffered file ([#858](https://github.com/gufo-org/gufo/issues/858)) ([abbc5cb](https://github.com/gufo-org/gufo/commit/abbc5cb8c3e72c76b35a2a371b1e0959b874442f))
* **qwen-flash:** bound draft-ID remapping and block selection by their buffers ([#849](https://github.com/gufo-org/gufo/issues/849)) ([81ff13c](https://github.com/gufo-org/gufo/commit/81ff13c3d394eb919d91074331ddd8d21701db45))
* **qwen-flash:** preserve YaRN in batched MTP cache catch-up ([#854](https://github.com/gufo-org/gufo/issues/854)) ([0f34adc](https://github.com/gufo-org/gufo/commit/0f34adcb8972979a3ea2160fea62bea41c7bb00a))
* **qwen38-flash-next:** run Q8_0 inject and alpha/beta weights on their F32 kernels ([#788](https://github.com/gufo-org/gufo/issues/788)) ([7b94a6a](https://github.com/gufo-org/gufo/commit/7b94a6a3fc52de630c15a822a0659a3e25f46bf7))
* **qwen38-flash-next:** start each qfn_mmq stream arena small and grow it in blocks ([#740](https://github.com/gufo-org/gufo/issues/740)) ([1a6655f](https://github.com/gufo-org/gufo/commit/1a6655fcb1c63f7e1e8aae7394e9c903b25ad91f))
* **qwen:** size the Q8_0 prefill GEMM activation scratch for its sum sidecar ([#777](https://github.com/gufo-org/gufo/issues/777)) ([2f10b15](https://github.com/gufo-org/gufo/commit/2f10b15d6deab4b19e7c45583ede120bb28dd2ed))


### Performance

* **qwen-flash:** keep the fused WMMA prefill attention past 262,144 visible tokens ([#866](https://github.com/gufo-org/gufo/issues/866)) ([538f6e4](https://github.com/gufo-org/gufo/commit/538f6e4dcb2d6a9d9574c9d6b001eea33c5e9c37))
* **qwen:** run split 9-15-row K-quant prefill GEMM calls as one masked 16-row pass ([#779](https://github.com/gufo-org/gufo/issues/779)) ([1fc829e](https://github.com/gufo-org/gufo/commit/1fc829eaef81a7fa26a3d3dfc3380c2eba80b99e))

## [0.26.0](https://github.com/gufo-org/gufo/compare/v0.25.0...v0.26.0) (2026-10-28)


### Features

* **embedding:** support native audio embeddings ([#819](https://github.com/gufo-org/gufo/issues/819)) ([1c62f0a](https://github.com/gufo-org/gufo/commit/1c62f0a9f96aec33c8128bf652eff312bdf56f90))
* **serve:** add llama-server's /apply-template ([#821](https://github.com/gufo-org/gufo/issues/821)) ([5efd8b7](https://github.com/gufo-org/gufo/commit/5efd8b772fc22a96adc9748db0c404434039fb00))
* **serve:** let --cache-ram-bytes turn the RAM cache off ([#783](https://github.com/gufo-org/gufo/issues/783)) ([62f0b96](https://github.com/gufo-org/gufo/commit/62f0b96a7b8ae37c8bfa08fcdca5c15989bf35dd))
* **serve:** select EmbeddingGemma 2 checkpoint prompts by name ([#823](https://github.com/gufo-org/gufo/issues/823)) ([ec4aacb](https://github.com/gufo-org/gufo/commit/ec4aacba29a497f72fb48d4fe246138f593911f5))


### Bug Fixes

* **audio:** drop resident safetensors pages after the TTS and ASR uploads ([#845](https://github.com/gufo-org/gufo/issues/845)) ([f1b6c54](https://github.com/gufo-org/gufo/commit/f1b6c54f1fe6145d40d93b820196974e6110840f))
* **build:** load the patched ROCr from build-tree HIP binaries ([#841](https://github.com/gufo-org/gufo/issues/841)) ([4a4a47c](https://github.com/gufo-org/gufo/commit/4a4a47c131890f417256c13e43e6adc925895ee7))
* **embedding:** accept MP3s without a Xing header and audio in llama.cpp multimodal_data ([#828](https://github.com/gufo-org/gufo/issues/828)) ([e472f83](https://github.com/gufo-org/gufo/commit/e472f8337161f2ca49b36a14c809777b645c0202))
* **qwen-flash:** remove lossy chunked DeltaNet prefill ([#850](https://github.com/gufo-org/gufo/issues/850)) ([c92b19c](https://github.com/gufo-org/gufo/commit/c92b19c5dc7ec12f83d0c2a5603cda78cc200b3e))
* **qwen:** keep model weights in device memory when the heap has room ([#708](https://github.com/gufo-org/gufo/issues/708)) ([5131f34](https://github.com/gufo-org/gufo/commit/5131f348c7372a4588f356ebb421b9be62ba2d7b))
* **rocm:** qualify image, H3, DeepSeek and Qwen models on 10.1 ([#768](https://github.com/gufo-org/gufo/issues/768)) ([c0e0be5](https://github.com/gufo-org/gufo/commit/c0e0be589d287b6935c3f8304e0bc3e63a9f05d5))
* **serve:** accept more than 128 function tools ([#827](https://github.com/gufo-org/gufo/issues/827)) ([b2fdc6a](https://github.com/gufo-org/gufo/commit/b2fdc6ac8ae2b2bca2810b362d1292179207d87f))
* **server:** drain the request before the loading responder closes ([#795](https://github.com/gufo-org/gufo/issues/795)) ([c811b45](https://github.com/gufo-org/gufo/commit/c811b4512a91279d9c62c370850541198d8c357e))


### Performance

* **qwen-flash:** faster Flash-Next prefill with chunked DeltaNet and stream overlap ([#758](https://github.com/gufo-org/gufo/issues/758)) ([538e2fd](https://github.com/gufo-org/gufo/commit/538e2fd7801901da712908ef69fdad510416dd7a))

## [0.25.0](https://github.com/gufo-org/gufo/compare/v0.24.0...v0.25.0) (2026-10-26)


### Features

* **server:** bound Qwen resolution for image_url.detail low ([#774](https://github.com/gufo-org/gufo/issues/774)) ([d8138a0](https://github.com/gufo-org/gufo/commit/d8138a01becd4a1e14bea546d113075c8f087ed2))
* **serve:** truncate over-length embedding inputs on request ([#765](https://github.com/gufo-org/gufo/issues/765)) ([d25b550](https://github.com/gufo-org/gufo/commit/d25b550b28372ba203fea040221d6cd7c2962fec))


### Bug Fixes

* **gcc16:** force-include &lt;format&gt; before HIP headers ([#782](https://github.com/gufo-org/gufo/issues/782)) ([fe9498d](https://github.com/gufo-org/gufo/commit/fe9498d10113201ef0e03fe04425922ca4e15cf6))
* **qwen:** fill device-resident weights from the shard file, not the mapping ([#794](https://github.com/gufo-org/gufo/issues/794)) ([2765d65](https://github.com/gufo-org/gufo/commit/2765d65e192e6db8bd51ee4f24c403d989f1ba5f))
* **windows:** bundle the LLVM OpenMP runtime ([#802](https://github.com/gufo-org/gufo/issues/802)) ([5eedd09](https://github.com/gufo-org/gufo/commit/5eedd09903be374596e630f686bb63aa030f12c4))
* **windows:** preserve concurrent cached PLE reads ([#805](https://github.com/gufo-org/gufo/issues/805)) ([f465f15](https://github.com/gufo-org/gufo/commit/f465f15f324cfba5ea3213667f472ecd06fc3157))


### Performance

* **embedding:** batch concurrent requests on the device ([#778](https://github.com/gufo-org/gufo/issues/778)) ([43deb1e](https://github.com/gufo-org/gufo/commit/43deb1ec1d37cb670f62a9a52fbac64ec281d0ce))
* **qwen:** keep a stable cache boundary after system instructions ([#773](https://github.com/gufo-org/gufo/issues/773)) ([1b8ad79](https://github.com/gufo-org/gufo/commit/1b8ad798a4bc443393754b6676ae6654a624dd0a))
* **qwen:** keep a stable cache boundary after system instructions REVERT ([#815](https://github.com/gufo-org/gufo/issues/815)) ([2479613](https://github.com/gufo-org/gufo/commit/2479613924ced6aab9b1411d0ba84fd00a87b9c7))

## [0.24.0](https://github.com/gufo-org/gufo/compare/v0.23.0...v0.24.0) (2026-10-25)


### Features

* **embeddings:** support image inputs ([#759](https://github.com/gufo-org/gufo/issues/759)) ([f928af5](https://github.com/gufo-org/gufo/commit/f928af54118718e88233c2b1fe608581ae100777))
* **serve:** accept repeated --served-model-name for one text model ([#736](https://github.com/gufo-org/gufo/issues/736)) ([67d1f6a](https://github.com/gufo-org/gufo/commit/67d1f6ac9ee17bb3aada02c83943bd577faab710))


### Bug Fixes

* **serve:** accept neutral llama.cpp sampler fields ([#732](https://github.com/gufo-org/gufo/issues/732)) ([68bfe92](https://github.com/gufo-org/gufo/commit/68bfe921016b9d94fb02f5e5104e84956e921945))
* **serve:** stop counting a connection once its response is sent ([#750](https://github.com/gufo-org/gufo/issues/750)) ([9c01c2f](https://github.com/gufo-org/gufo/commit/9c01c2f9592a10c00a69b4bf5cd876267183c4c0))


### Performance

* **embedding:** run EmbeddingGemma 2 attention on matrix cores ([#770](https://github.com/gufo-org/gufo/issues/770)) ([194db88](https://github.com/gufo-org/gufo/commit/194db882a857ee10c246a6a85cc787948a0028cf))


### Documentation

* **embeddings:** note client settings for LangChain and paperless-ngx ([#785](https://github.com/gufo-org/gufo/issues/785)) ([8c1eeb9](https://github.com/gufo-org/gufo/commit/8c1eeb98ca731af3c1a4d1b7a342e5f031d9a10c))

## [0.23.0](https://github.com/gufo-org/gufo/compare/v0.22.0...v0.23.0) (2026-10-24)


### Features

* **embedding:** serve EmbeddingGemma 2 on /v1/embeddings ([#700](https://github.com/gufo-org/gufo/issues/700)) ([de365fe](https://github.com/gufo-org/gufo/commit/de365fe5050944479db85afb8434ea19b2480e2d))
* **models:** read model weights through the mapped_file leaf ([#738](https://github.com/gufo-org/gufo/issues/738)) ([de43b8a](https://github.com/gufo-org/gufo/commit/de43b8a047735fb296a5d4f1218f1f628cbff5ac))
* **platform:** map GGUF shards through the mapped_file leaf ([#734](https://github.com/gufo-org/gufo/issues/734)) ([5c8bb39](https://github.com/gufo-org/gufo/commit/5c8bb390a40ef7d4482a1112d7b1e3c903003215))


### Performance

* **embedding:** tune GEMMs per shape with hipBLASLt ([#751](https://github.com/gufo-org/gufo/issues/751)) ([6fab9a2](https://github.com/gufo-org/gufo/commit/6fab9a24611b9551f7fa6a0e89bbffe7dd4d678f))
* **qwen-flash:** reduce selector spills and improve sampled drafting ([#742](https://github.com/gufo-org/gufo/issues/742)) ([d4eaaad](https://github.com/gufo-org/gufo/commit/d4eaaad438a498bf9ba541afe3d655f70dd4196d))
* **qwen38_flash_next:** let a lone greedy MTP request use the small draft head ([#733](https://github.com/gufo-org/gufo/issues/733)) ([cc5f405](https://github.com/gufo-org/gufo/commit/cc5f405a7322a5503f19c7d677c49b4e2ebc8351))
* **qwen38_flash_next:** let greedy requests with penalties or constraints use the small draft head ([#739](https://github.com/gufo-org/gufo/issues/739)) ([41d9346](https://github.com/gufo-org/gufo/commit/41d93463f829fdbef2eb1bb5d21658e080d00489))


### Documentation

* **agents:** describe where Windows portability code lives ([#754](https://github.com/gufo-org/gufo/issues/754)) ([f66dff9](https://github.com/gufo-org/gufo/commit/f66dff9ee18d06998c959cd5a0d547eb6cfcb26a))

## [0.22.0](https://github.com/gufo-org/gufo/compare/v0.21.0...v0.22.0) (2026-10-23)


### Features

* **platform:** add the net leaf ([#711](https://github.com/gufo-org/gufo/issues/711)) ([7c81965](https://github.com/gufo-org/gufo/commit/7c8196527ad02ac4c585a32b1131fab7e70509d0))
* **platform:** add the publish and prefault leaves ([#729](https://github.com/gufo-org/gufo/issues/729)) ([ddc31dc](https://github.com/gufo-org/gufo/commit/ddc31dc7685101b1d2336328a293b72fab0bed70))
* **platform:** add the shutdown_signal leaf ([#713](https://github.com/gufo-org/gufo/issues/713)) ([79c2dcc](https://github.com/gufo-org/gufo/commit/79c2dcc56f00c69d4085e52bde6d290b74a99ccd))


### Bug Fixes

* **cache:** free RAM before writing oversized checkpoints to disk ([#677](https://github.com/gufo-org/gufo/issues/677)) ([8a79230](https://github.com/gufo-org/gufo/commit/8a79230617490ff374b38edc002a8e68d241cf68))
* **qwen38-flash-next:** give each HIP stream its own qfn_mmq scratch arena ([#707](https://github.com/gufo-org/gufo/issues/707)) ([68ff41a](https://github.com/gufo-org/gufo/commit/68ff41a86f1e76bfd61ca85e47722ae78c17f05e))
* **qwen38-flash-next:** sum BF16 indexer projections in K order for every token ([#714](https://github.com/gufo-org/gufo/issues/714)) ([6c8b2ff](https://github.com/gufo-org/gufo/commit/6c8b2ff4c7d6c7c2e00835f84e6d4b4cee1c7226))
* **serve:** accept standard image detail values as automatic detail ([#722](https://github.com/gufo-org/gufo/issues/722)) ([9fa3645](https://github.com/gufo-org/gufo/commit/9fa36451facdf2b231d52afec72670a0fe2543b5))


### Performance

* **qwen38_flash_next:** price unseeded sampled MTP drafts at the decoding occupancy ([#703](https://github.com/gufo-org/gufo/issues/703)) ([1167a94](https://github.com/gufo-org/gufo/commit/1167a94475ad888534a99c9fb72401636cd9f7e7))

## [0.21.0](https://github.com/gufo-org/gufo/compare/v0.20.0...v0.21.0) (2026-10-22)


### Features

* **platform:** add the os_info leaf ([#709](https://github.com/gufo-org/gufo/issues/709)) ([3aa068b](https://github.com/gufo-org/gufo/commit/3aa068b419e6272482298c1c6daccafb122764fa))
* **qwen-flash:** opt-in YaRN context extension ([#350](https://github.com/gufo-org/gufo/issues/350)) ([2947624](https://github.com/gufo-org/gufo/commit/2947624b9e65f52d4b4fdd96297f39e040a0ed24))


### Bug Fixes

* **bench:** stop the benchmark server when startup is interrupted ([#702](https://github.com/gufo-org/gufo/issues/702)) ([53ebb51](https://github.com/gufo-org/gufo/commit/53ebb51ed0c0b24dbe7c1e35dce6c97d82740bfe))


### Performance

* **qwen:** reduce Flash-Next sampled decoding overhead ([#699](https://github.com/gufo-org/gufo/issues/699)) ([e060aee](https://github.com/gufo-org/gufo/commit/e060aeed923730b74ea3f4116c3c39de55741370))

## [0.20.0](https://github.com/gufo-org/gufo/compare/v0.19.0...v0.20.0) (2026-10-21)


### Features

* **cache:** add Qwen 27B AR and DFlash2 cache continuation ([#540](https://github.com/gufo-org/gufo/issues/540)) ([da5d5a2](https://github.com/gufo-org/gufo/commit/da5d5a2b237c654bc0523e33fdee10465240da0c))


### Bug Fixes

* **rocm:** let DeepSeek V4 Flash load when weights exceed the device pool ([#609](https://github.com/gufo-org/gufo/issues/609)) ([a7ca5f0](https://github.com/gufo-org/gufo/commit/a7ca5f0fc143203219dc71e079683d16642c6f15))
* **serve:** answer Expect: 100-continue before reading the body ([#686](https://github.com/gufo-org/gufo/issues/686)) ([f020bd3](https://github.com/gufo-org/gufo/commit/f020bd3a037d58aa42138ca11fa895524e43cdd6))
* **serve:** report no content for whitespace between parallel tool calls ([#676](https://github.com/gufo-org/gufo/issues/676)) ([9a9717c](https://github.com/gufo-org/gufo/commit/9a9717c1dae67a2623ba862b775bd9ba24633e0a))


### Performance

* **cache:** avoid repeated resets and full checkpoint lookups ([#680](https://github.com/gufo-org/gufo/issues/680)) ([887c38a](https://github.com/gufo-org/gufo/commit/887c38a17896d7c01095dfd0cf4b990378442846))
* **qwen-flash:** prefetch n-gram rows across cache checkpoints ([#683](https://github.com/gufo-org/gufo/issues/683)) ([62c840e](https://github.com/gufo-org/gufo/commit/62c840ee71cf203f2d1d0078f010bb0fec49b759))
* **qwen38_flash_next:** take the vector-load small projection path for F16 and BF16 weights ([#681](https://github.com/gufo-org/gufo/issues/681)) ([ae9ec69](https://github.com/gufo-org/gufo/commit/ae9ec6918350a97f78eeb680ae4ebf8fc0e9ad86))

## [0.19.0](https://github.com/gufo-org/gufo/compare/v0.18.0...v0.19.0) (2026-10-20)


### Features

* **serve:** report model, build, context, sessions and vision in /props ([#669](https://github.com/gufo-org/gufo/issues/669)) ([4862410](https://github.com/gufo-org/gufo/commit/4862410edc6852305e3b0f1b226332591fc56948))


### Bug Fixes

* **ci:** refresh the release app token after waiting for release checks ([#665](https://github.com/gufo-org/gufo/issues/665)) ([270b533](https://github.com/gufo-org/gufo/commit/270b53338829d25ea05dab633fb924621d9168f9))
* **ds4:** stop the batched greedy DSpark frontier before an accepted EOS ([#668](https://github.com/gufo-org/gufo/issues/668)) ([1b0a5a8](https://github.com/gufo-org/gufo/commit/1b0a5a845f18ad67fde0920d44785c2ffa1bd7d5))
* **qwen-flash:** store d*sum(q) as the q8_1 block sum ([#667](https://github.com/gufo-org/gufo/issues/667)) ([86730bd](https://github.com/gufo-org/gufo/commit/86730bda292fc15d8fbbb37cc89e1268431f3b48))


### Performance

* **qwen-flash:** let sampled requests use the small MTP draft head on multi-session servers ([#655](https://github.com/gufo-org/gufo/issues/655)) ([b6a671c](https://github.com/gufo-org/gufo/commit/b6a671c813f6ff35d9a2a0d2a57f59dad870c1a4))
* **qwen27b:** run verification attention rows &gt; 1 as wave64 with shared per-head steps ([#660](https://github.com/gufo-org/gufo/issues/660)) ([d1830de](https://github.com/gufo-org/gufo/commit/d1830de7df21815ace54ecc76df738ed04e7d5c7))
* **qwen38_flash_next:** price greedy MTP drafts at the decoding occupancy ([#664](https://github.com/gufo-org/gufo/issues/664)) ([2caf2a8](https://github.com/gufo-org/gufo/commit/2caf2a82f5620d8de77661cb64e5af1bed044a89))


### Documentation

* note Claude Code is unsupported with Qwen ([#672](https://github.com/gufo-org/gufo/issues/672)) ([8775189](https://github.com/gufo-org/gufo/commit/877518953c153b2aac57f7b77fc904e8540b1fde))

## [0.18.0](https://github.com/gufo-org/gufo/compare/v0.17.1...v0.18.0) (2026-10-19)


### Features

* **serve:** honor the Messages thinking budget ([#524](https://github.com/gufo-org/gufo/issues/524)) ([295f26c](https://github.com/gufo-org/gufo/commit/295f26ce63e93b288a47b1bb54d8d823a93e8c0d))
* **serve:** prepare the structured-output vocabulary before serving ([#638](https://github.com/gufo-org/gufo/issues/638)) ([e773094](https://github.com/gufo-org/gufo/commit/e77309421b0013d44d62ec0824be63c057a970e5))


### Bug Fixes

* **cache:** keep disk-cache snapshots out of the page cache ([#477](https://github.com/gufo-org/gufo/issues/477)) ([bff0eec](https://github.com/gufo-org/gufo/commit/bff0eec6f1eaaa5847d00712787153a3de78a4a0))
* **ci:** keep the devlab report when later jobs start ([#659](https://github.com/gufo-org/gufo/issues/659)) ([fbfc712](https://github.com/gufo-org/gufo/commit/fbfc712586c8972b2209bd6c13c69c8146311b81))
* **qwen-flash:** pin square-accumulation order in all HcCombineMoeF16Kernel instantiations ([#624](https://github.com/gufo-org/gufo/issues/624)) ([43b829d](https://github.com/gufo-org/gufo/commit/43b829d4dee5574d66fbd499640e56f40cbf4baa))
* **qwen:** reuse historical image preprocessing ([#645](https://github.com/gufo-org/gufo/issues/645)) ([f948ddd](https://github.com/gufo-org/gufo/commit/f948ddd371bf2a3f7360b7069fc703badbbc07ed))
* **serve:** reclaimable page cache is available in cgroup space ([#637](https://github.com/gufo-org/gufo/issues/637)) ([9e1afaa](https://github.com/gufo-org/gufo/commit/9e1afaaa4da0f36e3638ac88bf12b054d2a1114a))
* **tools:** preserve optional arguments in native calls ([#643](https://github.com/gufo-org/gufo/issues/643)) ([a802cc7](https://github.com/gufo-org/gufo/commit/a802cc77c795e26388e45cacec94aafcd69d12b4))


### Performance

* **flash-next:** reduce prompt and decode stalls ([#653](https://github.com/gufo-org/gufo/issues/653)) ([3462aa9](https://github.com/gufo-org/gufo/commit/3462aa96d0c02da3a6a511e0be197e186bd9f36a))
* **qwen:** fork the recurrent state across two buffers instead of copying it ([#650](https://github.com/gufo-org/gufo/issues/650)) ([95ba1ea](https://github.com/gufo-org/gufo/commit/95ba1eae63f62fd8a4576f6313ffa6be05f1e83b))

## [0.17.1](https://github.com/gufo-org/gufo/compare/v0.17.0...v0.17.1) (2026-10-18)


### Bug Fixes

* **cache:** reject capacity retries before reading payloads ([#636](https://github.com/gufo-org/gufo/issues/636)) ([778061d](https://github.com/gufo-org/gufo/commit/778061d453ff4cf863c47059f8ea63142c7f5ca1))
* **inference:** reduce Flash-Next cache and cold-prompt stalls ([#633](https://github.com/gufo-org/gufo/issues/633)) ([1f97092](https://github.com/gufo-org/gufo/commit/1f97092333af10f29887b098d9cb8a041610004a))


### Performance

* **qwen-flash:** copy repeated context into MTP drafts ([#613](https://github.com/gufo-org/gufo/issues/613)) ([8497711](https://github.com/gufo-org/gufo/commit/8497711eb154ecff55ef4119a01ace214149e276))
* **qwen-flash:** cut prefill and decode attention cost at depth ([#626](https://github.com/gufo-org/gufo/issues/626)) ([5a1669a](https://github.com/gufo-org/gufo/commit/5a1669a6c15129917a4e91b77bd36ddaa3e7b0d8))

## [0.17.0](https://github.com/gufo-org/gufo/compare/v0.16.0...v0.17.0) (2026-10-17)


### Features

* **serve:** prepare the Qwen vision sidecar before serving ([#616](https://github.com/gufo-org/gufo/issues/616)) ([0604e9e](https://github.com/gufo-org/gufo/commit/0604e9ebc41388ab610ed0bed198e49ef96a6a82))


### Bug Fixes

* **cache:** bound publication waits by observed prefill cost ([#621](https://github.com/gufo-org/gufo/issues/621)) ([e8e508f](https://github.com/gufo-org/gufo/commit/e8e508f51826c86d567ed3b7ea6bf2b1eaeb1ab5))
* **cache:** let disk eviction wait for startup orphan reclamation ([#625](https://github.com/gufo-org/gufo/issues/625)) ([78248de](https://github.com/gufo-org/gufo/commit/78248de23cc279b1a5d4443cef816a6059bde189))
* **serve:** keep the decimal spelling of integer JSON numbers ([#614](https://github.com/gufo-org/gufo/issues/614)) ([22e3676](https://github.com/gufo-org/gufo/commit/22e36769f9b25a57963a9215003eae0f5f9c15d6))
* **serve:** preserve cache reuse when tool choice changes ([#623](https://github.com/gufo-org/gufo/issues/623)) ([23d970d](https://github.com/gufo-org/gufo/commit/23d970d0ca0ef694dafec64131269f9c24590940))


### Performance

* **qwen-flash:** compact sparse attention selections once per row ([#622](https://github.com/gufo-org/gufo/issues/622)) ([baf5a1b](https://github.com/gufo-org/gufo/commit/baf5a1bbab5b99a8278c4869122fd173b205671b))

## [0.16.0](https://github.com/gufo-org/gufo/compare/v0.15.1...v0.16.0) (2026-10-16)


### Features

* **serve:** keep /health answering while the model and disk tier load ([#610](https://github.com/gufo-org/gufo/issues/610)) ([f6ca085](https://github.com/gufo-org/gufo/commit/f6ca085a5091eed23541cde6fa1ff282cbd4c68b))


### Performance

* **cache:** reduce long-context checkpoint overhead ([#615](https://github.com/gufo-org/gufo/issues/615)) ([f8be547](https://github.com/gufo-org/gufo/commit/f8be5474228095dd0eec6cec14cbc6ba7f070f90))
* **qwen-flash:** score MTP drafts over request vocabulary ([#603](https://github.com/gufo-org/gufo/issues/603)) ([51a8b00](https://github.com/gufo-org/gufo/commit/51a8b00d9dc5360688fc8642df94dfe1387ec2ca))


### Code Refactoring

* **serve:** move tool-call dialect parsing next to its renderers ([#592](https://github.com/gufo-org/gufo/issues/592)) ([b6ba33d](https://github.com/gufo-org/gufo/commit/b6ba33d48c7a2749a2e0b98e891a22e27d17cbdb))

## [0.15.1](https://github.com/gufo-org/gufo/compare/v0.15.0...v0.15.1) (2026-10-15)


### Bug Fixes

* **cache:** avoid artifact hashing for RAM-only Flash-Next ([#602](https://github.com/gufo-org/gufo/issues/602)) ([43ad9a5](https://github.com/gufo-org/gufo/commit/43ad9a5a7133dabc820a66a825247331c5503bf4))
* **cache:** repair flash-next functional cache regressions ([#596](https://github.com/gufo-org/gufo/issues/596)) ([ec69aa3](https://github.com/gufo-org/gufo/commit/ec69aa328ea78f00090c42a78034cc012eae151e))

## [0.15.0](https://github.com/gufo-org/gufo/compare/v0.14.0...v0.15.0) (2026-10-13)


### Features

* **serve:** accept IPv6 listen addresses ([#585](https://github.com/gufo-org/gufo/issues/585)) ([a690704](https://github.com/gufo-org/gufo/commit/a690704b99ce7f8b962e1b6004c07c7af24e0ce3))


### Bug Fixes

* **cache:** pin Flash-Next cache pages without compacting host memory ([#589](https://github.com/gufo-org/gufo/issues/589)) ([10c40b0](https://github.com/gufo-org/gufo/commit/10c40b0d09b6a58f0932c1f73e08318647b75474))
* **serve:** accept response schemas without additionalProperties ([#582](https://github.com/gufo-org/gufo/issues/582)) ([c406e40](https://github.com/gufo-org/gufo/commit/c406e40e97fcee008e47e3447b71f82066def6a7))


### Performance

* **qwen-flash:** speed up wide prefill kernels ([#571](https://github.com/gufo-org/gufo/issues/571)) ([a33e989](https://github.com/gufo-org/gufo/commit/a33e989cf49e533e3cb591a9f0483c67c590ba50))

## [0.14.0](https://github.com/gufo-org/gufo/compare/v0.13.0...v0.14.0) (2026-10-13)


### Features

* **cache:** wire Flash-Next serving to the component cache ([#573](https://github.com/gufo-org/gufo/issues/573)) ([058d6e8](https://github.com/gufo-org/gufo/commit/058d6e80c21089b8e3af76b6ae3c7b31a01d541f))
* **rocm:** managed-alloc fallback ([#460](https://github.com/gufo-org/gufo/issues/460)) ([248d902](https://github.com/gufo-org/gufo/commit/248d902e206be118216d8a3627718824816b0ab2))


### Documentation

* update readme ([#574](https://github.com/gufo-org/gufo/issues/574)) ([8d4e1a2](https://github.com/gufo-org/gufo/commit/8d4e1a2b74c94752b9b16b1dda153cbf52638443))

## [0.13.0](https://github.com/gufo-org/gufo/compare/v0.12.0...v0.13.0) (2026-10-12)


### Features

* **serve:** exit as device_lost when generation stops making progress ([#541](https://github.com/gufo-org/gufo/issues/541)) ([2b4daed](https://github.com/gufo-org/gufo/commit/2b4daede679e4fd46cc77e5658606306ed5ed2f0))


### Bug Fixes

* **serve:** accept a boolean or null stream on compatibility routes ([#544](https://github.com/gufo-org/gufo/issues/544)) ([ccf3463](https://github.com/gufo-org/gufo/commit/ccf3463f028bc97e5082cb2c612cf40da9cf77c4))
* **serve:** honor cache_prompt on raw completion routes ([#557](https://github.com/gufo-org/gufo/issues/557)) ([956f3d6](https://github.com/gufo-org/gufo/commit/956f3d62386a80d327aea9610310fef105c31b95))


### Performance

* **qwen-flash:** use 128 n-gram readers as documented ([#551](https://github.com/gufo-org/gufo/issues/551)) ([f37fb01](https://github.com/gufo-org/gufo/commit/f37fb012281bbd58054b1082933153bbec303106))


### Documentation

* **cache:** warn that the automatic RAM budget ignores later co-resident servers ([#556](https://github.com/gufo-org/gufo/issues/556)) ([113d42f](https://github.com/gufo-org/gufo/commit/113d42fbfecd57b3b18126a22e62a04dd4a4b51d))
* **qwen-flash:** record rejected fork prefill kernels ([#565](https://github.com/gufo-org/gufo/issues/565)) ([1834991](https://github.com/gufo-org/gufo/commit/1834991317745b095ea42fc21370d594fabd519c))

## [0.12.0](https://github.com/gufo-org/gufo/compare/v0.11.0...v0.12.0) (2026-10-11)


### Features

* **cache:** add Flash-Next continuation adapter ([#536](https://github.com/gufo-org/gufo/issues/536)) ([95d7619](https://github.com/gufo-org/gufo/commit/95d76191e9d2cd434bc13defb971bf26e3c9e3f0))
* **serve:** accept image blocks in Messages user turns and tool results ([#510](https://github.com/gufo-org/gufo/issues/510)) ([6a1b136](https://github.com/gufo-org/gufo/commit/6a1b136aba5982e5a35c628824f2380280766d46))
* **serve:** implement /v1/messages/count_tokens ([#512](https://github.com/gufo-org/gufo/issues/512)) ([8ca0c4a](https://github.com/gufo-org/gufo/commit/8ca0c4a0b06310b522cb045a0ed820e635204845))


### Bug Fixes

* **json:** keep mixed string and number enum values reachable ([#535](https://github.com/gufo-org/gufo/issues/535)) ([d8efa54](https://github.com/gufo-org/gufo/commit/d8efa542cf5a991bf5fd293cba2602e8605529ac))
* **qwen:** keep Flash-Next snapshots within device headroom ([#549](https://github.com/gufo-org/gufo/issues/549)) ([eada056](https://github.com/gufo-org/gufo/commit/eada0564b000a8e630118d10dbdd5bfcd6dc13d7))
* **serve:** exclude cache reads from Messages input_tokens ([#530](https://github.com/gufo-org/gufo/issues/530)) ([2a3b09e](https://github.com/gufo-org/gufo/commit/2a3b09e187208c3895eee7391ca0a6a3179a2efc))
* **serve:** keep listening WebSocket clients connected past the idle timeout ([#527](https://github.com/gufo-org/gufo/issues/527)) ([15ecb0c](https://github.com/gufo-org/gufo/commit/15ecb0c2ce5d98e49423a2cab7eb148754488d44))
* **serve:** name max_completion_tokens in its Chat validation error ([#545](https://github.com/gufo-org/gufo/issues/545)) ([fc760f2](https://github.com/gufo-org/gufo/commit/fc760f2075cf6b812089794b69feaa66597cd8af))
* **serve:** refuse images before reading when model is text-only ([#550](https://github.com/gufo-org/gufo/issues/550)) ([3323e7c](https://github.com/gufo-org/gufo/commit/3323e7c789e7cdf56466602e032e24e8c01df0a8))
* **serve:** reject HTTP header names that are not tokens ([#533](https://github.com/gufo-org/gufo/issues/533)) ([da827bd](https://github.com/gufo-org/gufo/commit/da827bd244fe277913bd9f98801b0117e02748ac))
* **serve:** report non-string models and documented reasoning efforts in Chat errors ([#546](https://github.com/gufo-org/gufo/issues/546)) ([8fa2369](https://github.com/gufo-org/gufo/commit/8fa2369f3de015a2281f86428b4b1eac30078c50))
* **serve:** stream native tool-call arguments incrementally ([#534](https://github.com/gufo-org/gufo/issues/534)) ([0377339](https://github.com/gufo-org/gufo/commit/037733911b8526b45ddcdf35df96508eea4fe25b))


### Documentation

* **cache:** note the final user turn an unchanged retry prefills ([#552](https://github.com/gufo-org/gufo/issues/552)) ([d03c8fc](https://github.com/gufo-org/gufo/commit/d03c8fcc8309e949357f9bba8d6a5fc1a1eebd3f))
* **server:** allow top_p zero and scope cache_prompt to its endpoints ([#548](https://github.com/gufo-org/gufo/issues/548)) ([d0f7c3a](https://github.com/gufo-org/gufo/commit/d0f7c3a023a11af81161264eeb24e9a97aa7498a))

## [0.11.0](https://github.com/gufo-org/gufo/compare/v0.10.0...v0.11.0) (2026-10-10)


### Features

* **cache:** add committed HIP backing and transfer pools ([#504](https://github.com/gufo-org/gufo/issues/504)) ([e6c850d](https://github.com/gufo-org/gufo/commit/e6c850d3df61578bf878db2ad1fc96266d8bc07d))
* **cache:** define disk manifests and metadata startup index ([#507](https://github.com/gufo-org/gufo/issues/507)) ([9024101](https://github.com/gufo-org/gufo/commit/9024101b12a59ffe8e067dfad25250236f57d1b9))
* **cache:** preserve borrowed rows while slots are idle ([#505](https://github.com/gufo-org/gufo/issues/505)) ([525d9c4](https://github.com/gufo-org/gufo/commit/525d9c4dbd4b126ed092bad2c736382392a8a4a1))
* **cache:** publish checkpoints with crash-safe recovery ([#509](https://github.com/gufo-org/gufo/issues/509)) ([d65addf](https://github.com/gufo-org/gufo/commit/d65addfb0ea246726640f1d149c6960fca5adbb9))
* **cache:** select and evict resident and durable checkpoints ([#517](https://github.com/gufo-org/gufo/issues/517)) ([92aaed5](https://github.com/gufo-org/gufo/commit/92aaed5d30cd82e5730e43be3501775b77c26e5e))
* **cache:** stream checkpoints through bounded transfer pieces ([#515](https://github.com/gufo-org/gufo/issues/515)) ([081e8a0](https://github.com/gufo-org/gufo/commit/081e8a048afe6611517fcac86c6c2e759f45e021))


### Bug Fixes

* **cache:** keep a live frontier when a branch restores its borrowed checkpoint ([#476](https://github.com/gufo-org/gufo/issues/476)) ([7def5ad](https://github.com/gufo-org/gufo/commit/7def5adcb8b2e13262eeed8bfc98d81a040cd84f))
* **ci:** match GitHub anchors and relative links in the docs check ([#528](https://github.com/gufo-org/gufo/issues/528)) ([b869dd0](https://github.com/gufo-org/gufo/commit/b869dd056f6d068839858c96eb34c20d2479fcf7))
* **ci:** require exact dependency inventory rows ([#531](https://github.com/gufo-org/gufo/issues/531)) ([3f53952](https://github.com/gufo-org/gufo/commit/3f53952e9ee68e3a64828a752d7bbac322135a58))
* **gguf:** decode 8- and 16-bit integer metadata arrays ([#532](https://github.com/gufo-org/gufo/issues/532)) ([3fac1bb](https://github.com/gufo-org/gufo/commit/3fac1bb51c97ef3606bf0bc662bf778344414cb4))
* **qwen-image:** stage weight uploads through pinned buffers ([#499](https://github.com/gufo-org/gufo/issues/499)) ([eb91daa](https://github.com/gufo-org/gufo/commit/eb91daa2b8ae2b2e35c11dc26c1cb08becb972cf))
* **serve:** log the cause of generic generation failures ([#520](https://github.com/gufo-org/gufo/issues/520)) ([b0af357](https://github.com/gufo-org/gufo/commit/b0af357b34a7b34055a155eb77168a4b10d6b0b2))
* **serve:** preserve images in tool results ([#506](https://github.com/gufo-org/gufo/issues/506)) ([5f0b194](https://github.com/gufo-org/gufo/commit/5f0b19403b1b9cecb018db2c39b9272daf6c0ab2))


### Performance

* **qwen-flash:** faster decode kernels ([#518](https://github.com/gufo-org/gufo/issues/518)) ([7e4621c](https://github.com/gufo-org/gufo/commit/7e4621c37079ab1bc62c1fd07ca03c37b85831cc))


### Documentation

* **cache:** mark implemented cards done in the card index ([#526](https://github.com/gufo-org/gufo/issues/526)) ([a41e88d](https://github.com/gufo-org/gufo/commit/a41e88da229441f31b49cef5ea1cbfc5f2704dc3))


### Code Refactoring

* **cache:** add coherent prefix index and lookup ([#500](https://github.com/gufo-org/gufo/issues/500)) ([e17e2d1](https://github.com/gufo-org/gufo/commit/e17e2d1dd1d05c0be14f0e9cc309a82f510de68d))
* **cache:** add common adapter contracts and fake adapter ([#493](https://github.com/gufo-org/gufo/issues/493)) ([c875a5b](https://github.com/gufo-org/gufo/commit/c875a5bd37eb390b1e08383459882085ce193a3b))
* **cache:** add resource ledger and transactional reservations ([#496](https://github.com/gufo-org/gufo/issues/496)) ([c8d3ab5](https://github.com/gufo-org/gufo/commit/c8d3ab550d20df3e5b3d244318e8c4757625dd5a))
* **cache:** add shared chunks and checkpoint provenance ([#498](https://github.com/gufo-org/gufo/issues/498)) ([087c192](https://github.com/gufo-org/gufo/commit/087c192d9f74b8557646699d6fc55bc23b37f25e))
* **cache:** add slot leases and transactional mutation guards ([#501](https://github.com/gufo-org/gufo/issues/501)) ([7035ebb](https://github.com/gufo-org/gufo/commit/7035ebbf70489bdc469dc1c701fea5295757297e))
* **cache:** port retention policy and replayable events ([#503](https://github.com/gufo-org/gufo/issues/503)) ([51ceefc](https://github.com/gufo-org/gufo/commit/51ceefc1989cb8689008628f64c1d3b988df7d5c))

## [0.10.0](https://github.com/gufo-org/gufo/compare/v0.9.1...v0.10.0) (2026-10-09)


### Features

* **serve:** support tools and streaming on /v1/messages ([#469](https://github.com/gufo-org/gufo/issues/469)) ([50900eb](https://github.com/gufo-org/gufo/commit/50900eb717925cc8e85ed952e1a75e522958a7f6))


### Performance

* **ds4:** keep the 8-row IQ2 gate/up tiles out of scratch ([#482](https://github.com/gufo-org/gufo/issues/482)) ([e9b0f0b](https://github.com/gufo-org/gufo/commit/e9b0f0b34b41b1968dcf6eb2f215cbfd53ae4ebe))
* **qwen-flash:** accelerate prefill and MTP catch-up ([#485](https://github.com/gufo-org/gufo/issues/485)) ([fd747a5](https://github.com/gufo-org/gufo/commit/fd747a51951ccd09eda20de5c34313670dc9b9d3))
* **qwen-image:** unroll the fused attention value halves ([#481](https://github.com/gufo-org/gufo/issues/481)) ([b2438a4](https://github.com/gufo-org/gufo/commit/b2438a4a12717cd0970b46bb293875eaf31d9839))


### Documentation

* **cache:** add cache redesign implementation cards ([#489](https://github.com/gufo-org/gufo/issues/489)) ([d221a01](https://github.com/gufo-org/gufo/commit/d221a01c74052737a04148ef86a47e1db3e270b7))
* **cache:** propose continuation cache redesign RFC ([#488](https://github.com/gufo-org/gufo/issues/488)) ([b3facce](https://github.com/gufo-org/gufo/commit/b3facce1180e4bb44ac25c8f8a0fee348ae36da9))

## [0.9.1](https://github.com/gufo-org/gufo/compare/v0.9.0...v0.9.1) (2026-10-08)


### Bug Fixes

* **cache:** drop the fixed 1 GiB cap on automatic disk staging ([#473](https://github.com/gufo-org/gufo/issues/473)) ([b39c530](https://github.com/gufo-org/gufo/commit/b39c530e70e87f4340e2230a155fd16d066d19f3))


### Performance

* **hip:** keep prefill GEMMs spill-free on clang 23 ([#459](https://github.com/gufo-org/gufo/issues/459)) ([f17e37b](https://github.com/gufo-org/gufo/commit/f17e37b8bb7df5fb83ea7ce6d4dc4ef6d6677253))
* **qwen-flash:** 4096-token prefill chunks with an overlapped n-gram gather ([#470](https://github.com/gufo-org/gufo/issues/470)) ([47b6391](https://github.com/gufo-org/gufo/commit/47b639159315fcdba17e6a144d67e273de9ead6e))


### Code Refactoring

* **qwen:** centralize control-token literals in one header ([#464](https://github.com/gufo-org/gufo/issues/464)) ([7701ba7](https://github.com/gufo-org/gufo/commit/7701ba769f454a580ffcbaa29538d65e7f33b38c))

## [0.9.0](https://github.com/gufo-org/gufo/compare/v0.8.1...v0.9.0) (2026-10-07)


### Features

* **server:** add opt-in --trace for text request content ([#458](https://github.com/gufo-org/gufo/issues/458)) ([ba97a2b](https://github.com/gufo-org/gufo/commit/ba97a2b33f3ab0ca3864b9b7b51c62dff199268f))


### Bug Fixes

* **cache:** exclude free CMA pages from the automatic RAM budget ([88d2139](https://github.com/gufo-org/gufo/commit/88d213943e4f97c292f1561ccae32bb42e2b35c3))
* **cache:** keep learned branch points under RAM pressure ([#466](https://github.com/gufo-org/gufo/issues/466)) ([7da04cd](https://github.com/gufo-org/gufo/commit/7da04cd37f70eb1c455fae7ceb2306f0cbbf4173))
* **serve:** default the per-client queue cap to --max-pending ([#467](https://github.com/gufo-org/gufo/issues/467)) ([04205d4](https://github.com/gufo-org/gufo/commit/04205d44611fdccca2be0ae70e685894fbd0d19a))
* **server:** hoist mid-conversation system messages to the leading block ([#449](https://github.com/gufo-org/gufo/issues/449)) ([05688f9](https://github.com/gufo-org/gufo/commit/05688f94aec4ecae21ea9be0d9eb5c6516e18829))


### Performance

* **qwen-flash:** reduce prefill work at long context ([#463](https://github.com/gufo-org/gufo/issues/463)) ([33d1b20](https://github.com/gufo-org/gufo/commit/33d1b208a72e9664fdc1267ae1c0d465cfb9b8f9))
* **qwen:** avoid flash-next prompt checkpoint copies ([#445](https://github.com/gufo-org/gufo/issues/445)) ([82711d8](https://github.com/gufo-org/gufo/commit/82711d8c8ceb8bda9024914b6e7ac8495ba41742))

## [0.8.1](https://github.com/gufo-org/gufo/compare/v0.8.0...v0.8.1) (2026-10-06)


### Bug Fixes

* **server:** keep tool calls in native model syntax ([#441](https://github.com/gufo-org/gufo/issues/441)) ([21d6e64](https://github.com/gufo-org/gufo/commit/21d6e64f137f6bcbc5a8bf63f900cab648188df7))
* **server:** remove leading answer blank lines after reasoning ([#446](https://github.com/gufo-org/gufo/issues/446)) ([4ec92f2](https://github.com/gufo-org/gufo/commit/4ec92f2d48bf594503f06917a5d96e9da2978aa7))
* **vision:** accept image histories within model context ([#447](https://github.com/gufo-org/gufo/issues/447)) ([167bad6](https://github.com/gufo-org/gufo/commit/167bad69e8ac45f0fdf4bb376959a10caca2844c))

## [0.8.0](https://github.com/gufo-org/gufo/compare/v0.7.1...v0.8.0) (2026-10-05)


### Features

* OpenAI Responses compatibility ([#434](https://github.com/gufo-org/gufo/issues/434)) ([d921a4b](https://github.com/gufo-org/gufo/commit/d921a4bd956424241e3e050cf981023b5b81e475))

## [0.7.1](https://github.com/gufo-org/gufo/compare/v0.7.0...v0.7.1) (2026-10-05)


### Bug Fixes

* **cache:** keep the stable checkpoint before replaced trailing user context ([#407](https://github.com/gufo-org/gufo/issues/407)) ([3d73237](https://github.com/gufo-org/gufo/commit/3d732377e5ac257fa3f1a59f57a17163e65763b5))
* **serve:** accept the reasoning fields Claude Code sends to /v1/messages ([#405](https://github.com/gufo-org/gufo/issues/405)) ([56383be](https://github.com/gufo-org/gufo/commit/56383be0718ea0ce203fc42a55581304f0bbc253))
* **serve:** preserve active disk-cache writes during startup ([#392](https://github.com/gufo-org/gufo/issues/392)) ([207bb4e](https://github.com/gufo-org/gufo/commit/207bb4e0bf6cd0c666674794ec1f0dcf0eb49cb9))
* **serve:** reclaim output capacity from abandoned streams ([#394](https://github.com/gufo-org/gufo/issues/394)) ([d85fb0f](https://github.com/gufo-org/gufo/commit/d85fb0fcd3fb906ea845f6e1404222d686b74e07))
* **server:** end DeepSeek tool output after the call block ([#397](https://github.com/gufo-org/gufo/issues/397)) ([0df6ba3](https://github.com/gufo-org/gufo/commit/0df6ba3e1b5eade79782aacac559a02ea5468c2b))
* **server:** keep DeepSeek client markup as content, as llama.cpp does ([#420](https://github.com/gufo-org/gufo/issues/420)) ([a72fc1e](https://github.com/gufo-org/gufo/commit/a72fc1e58c8d4784e7b824b4aad93a76b5a197b1))
* **server:** keep tool-call framing out of assistant content ([#400](https://github.com/gufo-org/gufo/issues/400)) ([d910b92](https://github.com/gufo-org/gufo/commit/d910b92f9d722ee4a0ed8770c16749dea472afe2))
* **server:** preserve literal reasoning tags when thinking is disabled ([#391](https://github.com/gufo-org/gufo/issues/391)) ([6c0d493](https://github.com/gufo-org/gufo/commit/6c0d493eb3cbcbc8571648260d952a3198fab784))


### Performance

* **qwen-flash:** faster prefill projections, attention and indexer ([#421](https://github.com/gufo-org/gufo/issues/421)) ([653a318](https://github.com/gufo-org/gufo/commit/653a31830447be5068a448a0284e43669d52f436))
* **sampling:** skip vocabulary blocks that cannot change the selected tokens ([#415](https://github.com/gufo-org/gufo/issues/415)) ([b945d0a](https://github.com/gufo-org/gufo/commit/b945d0afdb26e4b790b8cb260762105e167ee1a3))


### Code Refactoring

* **serve:** keep Messages reasoning parsers together ([#428](https://github.com/gufo-org/gufo/issues/428)) ([c3f4a9c](https://github.com/gufo-org/gufo/commit/c3f4a9c26031b04f34b84a96f4d8cfc8f4337ad2))

## [0.7.0](https://github.com/gufo-org/gufo/compare/v0.6.0...v0.7.0) (2026-10-04)


### Features

* **serve:** learn shared-prefix boundaries in the RAM cache ([#386](https://github.com/gufo-org/gufo/issues/386)) ([2ba3be2](https://github.com/gufo-org/gufo/commit/2ba3be24039186e00d2cecdb969f99ad695e1161))
* **serve:** let an explicit RAM cache limit exceed the automatic budget ([#384](https://github.com/gufo-org/gufo/issues/384)) ([53c5906](https://github.com/gufo-org/gufo/commit/53c590649295edf63abcc06117231cdc890b69c7))
* **serve:** report live sessions and llama.cpp-compatible metrics ([#389](https://github.com/gufo-org/gufo/issues/389)) ([1b4e682](https://github.com/gufo-org/gufo/commit/1b4e6825f2cf6fa2203af3b4b3596bf25e4bfa4e))
* **server:** export speculative verification round metrics ([#403](https://github.com/gufo-org/gufo/issues/403)) ([8bdde80](https://github.com/gufo-org/gufo/commit/8bdde807e57fadfe57f4a1005707559ae6afc82f))


### Bug Fixes

* **server:** detect idle GPU loss and defer streaming success ([#406](https://github.com/gufo-org/gufo/issues/406)) ([6a9ea9f](https://github.com/gufo-org/gufo/commit/6a9ea9f263de4598a4c49954f10c5d285bcf7635))
* **server:** parse tool output using the admitted request format ([#393](https://github.com/gufo-org/gufo/issues/393)) ([c33e050](https://github.com/gufo-org/gufo/commit/c33e050eced6389852617994fe7349367df4c900))
* **server:** reuse replayed tool turns with union and typed arguments ([#404](https://github.com/gufo-org/gufo/issues/404)) ([ea06418](https://github.com/gufo-org/gufo/commit/ea064189976242f33f54bac92be6e0cdbd33fc48))

## [0.6.0](https://github.com/gufo-org/gufo/compare/v0.5.0...v0.6.0) (2026-10-03)


### Features

* **serve:** share in-flight prefixes between concurrent requests ([#382](https://github.com/gufo-org/gufo/issues/382)) ([c1eba3d](https://github.com/gufo-org/gufo/commit/c1eba3de7ffc1f71c9ad6e395138f9bf89ddcc7b))


### Bug Fixes

* **serve:** exit and report device_lost when the GPU context is lost ([#390](https://github.com/gufo-org/gufo/issues/390)) ([ee2bff3](https://github.com/gufo-org/gufo/commit/ee2bff34d8cbb95a29f8abe85e046dd49382b5c9))
* **server:** improve error messaging on streaming generation failure ([#385](https://github.com/gufo-org/gufo/issues/385)) ([bf60539](https://github.com/gufo-org/gufo/commit/bf605399477b694c19adb6313952ecbf0986d8be))
* **server:** preserve JSON string ownership during tool recovery ([#396](https://github.com/gufo-org/gufo/issues/396)) ([2c6a106](https://github.com/gufo-org/gufo/commit/2c6a1064f39a4d3ea0b8d92efea0beedf18150f1))
* **serve:** separate Messages thinking blocks and accept the thinking field ([#380](https://github.com/gufo-org/gufo/issues/380)) ([b27f1ec](https://github.com/gufo-org/gufo/commit/b27f1ec0189e5410028029f249915dfa8813b749))

## [0.5.0](https://github.com/gufo-org/gufo/compare/v0.4.0...v0.5.0) (2026-10-02)


### Features

* **server:** accept WebP and other spellings of image data URLs ([#352](https://github.com/gufo-org/gufo/issues/352)) ([d707143](https://github.com/gufo-org/gufo/commit/d707143223c52b91da3f0fb231ba85ac3eab247c))


### Bug Fixes

* **cache:** keep cache reuse advancing as conversations grow ([#358](https://github.com/gufo-org/gufo/issues/358)) ([9afdd48](https://github.com/gufo-org/gufo/commit/9afdd4802fbe4f4afeb5607a8cc4da905f8e4d70))
* **cache:** keep cached conversations independent of execution sessions ([#369](https://github.com/gufo-org/gufo/issues/369)) ([d7cf7ee](https://github.com/gufo-org/gufo/commit/d7cf7ee4b8e3a506ee1ec9c301e8a3efd10c6c94))
* **cache:** retain prefixes across conversation history edits ([#362](https://github.com/gufo-org/gufo/issues/362)) ([68e8475](https://github.com/gufo-org/gufo/commit/68e8475dbdf9dcddbe0a3ea645646b453b0bb89c))
* **qwen-image:** prevent source-noise reuse in image edits ([#377](https://github.com/gufo-org/gufo/issues/377)) ([1071b36](https://github.com/gufo-org/gufo/commit/1071b361eb69e561125543382ebbf3f212c2cc5a))
* **server:** advertise loaded model input modalities ([#367](https://github.com/gufo-org/gufo/issues/367)) ([5451525](https://github.com/gufo-org/gufo/commit/54515255300de99a8cbbae192d25b42959b5c973))
* **server:** keep literal tool markers inside constrained reasoning ([#361](https://github.com/gufo-org/gufo/issues/361)) ([2ab0c4b](https://github.com/gufo-org/gufo/commit/2ab0c4b70ffec6be8c306a6ed299acef76c903e7))
* **server:** preserve native tool schemas and historical calls ([#373](https://github.com/gufo-org/gufo/issues/373)) ([594a623](https://github.com/gufo-org/gufo/commit/594a623913b4109e4499885e9f73ed4d4ad3698e))


### Performance

* **serve:** skip checkpoints that barely advance a prefix ([#348](https://github.com/gufo-org/gufo/issues/348)) ([93af45d](https://github.com/gufo-org/gufo/commit/93af45d48a712d2c14105ef5a9347942e2260f14))

## [0.4.0](https://github.com/gufo-org/gufo/compare/v0.3.0...v0.4.0) (2026-10-01)


### Features

* **serve:** expose live token and request metrics ([#351](https://github.com/gufo-org/gufo/issues/351)) ([03d7c72](https://github.com/gufo-org/gufo/commit/03d7c727c48c62960549ef5f27cd2ea9897c873e))
* **serve:** report cache eviction and retained snapshot capacity ([#353](https://github.com/gufo-org/gufo/issues/353)) ([bda078c](https://github.com/gufo-org/gufo/commit/bda078c3c3b087bd4bf5a9b09f4fd38653eb2e94))
* **server:** make log verbosity configurable with --log-level ([#319](https://github.com/gufo-org/gufo/issues/319)) ([6a32726](https://github.com/gufo-org/gufo/commit/6a32726068f21f94db0761edba1da00eefe99c4b))
* **server:** support llama-server return_progress on streaming completions ([#344](https://github.com/gufo-org/gufo/issues/344)) ([741722b](https://github.com/gufo-org/gufo/commit/741722b04eb474b3d75c7e2c0a4d727ae21f6577))


### Bug Fixes

* **qwen-flash:** preserve seeded MTP replay across cache reuse ([#330](https://github.com/gufo-org/gufo/issues/330)) ([a917b79](https://github.com/gufo-org/gufo/commit/a917b790df8d5fd98205abddd3b7c1afd0ca9458))
* **serve:** preserve native tool calls during constrained decoding ([#324](https://github.com/gufo-org/gufo/issues/324)) ([b26de0d](https://github.com/gufo-org/gufo/commit/b26de0d30caa363cc6694bddd094af9bd88ec62b))
* **server:** keep SSE streams alive during generation ([#334](https://github.com/gufo-org/gufo/issues/334)) ([c6e1069](https://github.com/gufo-org/gufo/commit/c6e10690c7532a99257b58319ca619a52351ddca))


### Performance

* **qwen-flash:** accelerate greedy penalties and fix sampling ranges ([#332](https://github.com/gufo-org/gufo/issues/332)) ([7e450e8](https://github.com/gufo-org/gufo/commit/7e450e8c0bc5458d056b34675e6d7f844d0ef23f))


### Documentation

* explain how the KV cache works in gufo ([#360](https://github.com/gufo-org/gufo/issues/360)) ([446cb14](https://github.com/gufo-org/gufo/commit/446cb141f8b312befdbfb58945c01db3897ba09a))

## [0.3.0](https://github.com/gufo-org/gufo/compare/v0.2.0...v0.3.0) (2026-09-30)


### Features

* **cli:** print startup banner on interactive commands ([#323](https://github.com/gufo-org/gufo/issues/323)) ([f783fed](https://github.com/gufo-org/gufo/commit/f783fedb9bea2ec7de941f6da4e02f4a4596b29e))


### Documentation

* update readme with link to gufo forks ([#327](https://github.com/gufo-org/gufo/issues/327)) ([8eedee6](https://github.com/gufo-org/gufo/commit/8eedee6fd904b8e6812f740f777fe84940f341c5))

## [0.2.0](https://github.com/gufo-org/gufo/compare/v0.1.1...v0.2.0) (2026-09-29)


### Features

* **sampling:** use official text-model defaults ([#282](https://github.com/gufo-org/gufo/issues/282)) ([eb91584](https://github.com/gufo-org/gufo/commit/eb915840ffb62a8ec4b5c1adb41b04b5c1c75892))


### Bug Fixes

* **serve:** bound hardware compute queues per server ([#317](https://github.com/gufo-org/gufo/issues/317)) ([9da89d6](https://github.com/gufo-org/gufo/commit/9da89d64b03c13f76085f6221074e927f9d91472))
* **server:** accept dotted and namespaced tool names ([#314](https://github.com/gufo-org/gufo/issues/314)) ([fee9d2a](https://github.com/gufo-org/gufo/commit/fee9d2a4c17ea2ff43d36672d9e693029bb810d4))

## [0.1.1](https://github.com/gufo-org/gufo/compare/v0.1.0...v0.1.1) (2026-09-28)


### Bug Fixes

* **server:** handle repeated tool-call parameters ([30392d5](https://github.com/gufo-org/gufo/commit/30392d5bbe96dc925f81e955d9e0285f4351ff34))

## [0.1.0] - 2026-09-28

Initial public development release for AMD Strix Halo (`gfx1151`).

### Features

- Native text inference and OpenAI-compatible serving for Qwen3.8 27B,
  Qwen3.8 Flash-Next and DeepSeek V4 Flash.
- Speech recognition with Qwen3-ASR and speech synthesis and voice cloning
  with Qwen3-TTS.
- Image generation with Qwen-Image-2.1 and experimental MiniMax H3
  video/audio generation.
- Continuous batching, request cancellation and conversation caching as
  first-class serving workloads.
- Reproducible Nix and CMake production builds specialized for Strix Halo.

### Performance

- Model-owned HIP kernels and speculative decoding paths for DFlash2, MTP and
  DSpark, with published matched quality and performance reports.

[0.1.0]: https://github.com/gufo-org/gufo/releases/tag/v0.1.0
