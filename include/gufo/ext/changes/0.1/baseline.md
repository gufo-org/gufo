# Source API baseline

Initial development API: metadata/registration, core capability kinds, typed
provider options and a borrowed logging Host handle. Compatibility requires
an `api_version` floor in the active compatibility series; the minor series must match and the requested
revision must be no newer than the runtime. An additive revision keeps older
extensions compatible within this series. No stable model/session ABI is
promised by this initial source API.
