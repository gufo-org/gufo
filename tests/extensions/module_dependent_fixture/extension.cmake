# An extension that depends on the in-tree qwen38_flash_next and links its
# exports and the Qwen family library; the module CLI test builds it as a
# module that takes their code from the gufo executable.
gufo_extension_target(NAME module_dependent_fixture_registration SOURCES registration.cpp
  LINK_LIBRARIES gufo_ext_runtime)
gufo_extension_target(NAME module_dependent_fixture_provider SOURCES provider.cpp
  LINK_LIBRARIES gufo_ext_runtime gufo_ext_core gufo::qwen38_flash_next gufo::qwen_family)
# Classes shared across the boundary: the executable's runner calls the
# module's Stages and WeightSource through their vtables.
gufo_extension_target(NAME module_dependent_fixture_hip SOURCES hip_provider.cpp
  LINK_LIBRARIES gufo_ext_runtime gufo_ext_core gufo_ext_hip gufo::qwen38_flash_next
  gufo::qwen38_flash_next_hip gufo::qwen_family)
if(TARGET module_dependent_fixture_hip)
  target_compile_definitions(module_dependent_fixture_hip PRIVATE ENGINE_ENABLE_HIP=1)
endif()
