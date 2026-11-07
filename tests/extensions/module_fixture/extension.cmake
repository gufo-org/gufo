# A standalone extension that `gufo extension install` builds as a module in
# the module CLI test.
gufo_extension_target(NAME module_fixture_registration SOURCES registration.cpp
  LINK_LIBRARIES gufo_ext_runtime)
gufo_extension_target(NAME module_fixture_provider SOURCES provider.cpp
  LINK_LIBRARIES gufo_ext_runtime gufo_ext_core)
