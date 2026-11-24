# MSVC's vectorized find rejects aggregate types in the SDK 0.3 importer.
# Use the same STL setting for every C++ target in the Loom runtime DLL.
add_compile_definitions($<$<COMPILE_LANGUAGE:CXX>:_USE_STD_VECTOR_ALGORITHMS=0>)
