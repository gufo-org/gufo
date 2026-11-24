# Resolve the prefix at install time, including cmake --install --prefix.
# A configure-time prefix would silently copy into Program Files instead.
if(WIN32)
  install(CODE [[
    execute_process(COMMAND "${CMAKE_COMMAND}" -E copy
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gufo.exe"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gufo-server.exe"
      COMMAND_ERROR_IS_FATAL ANY)
  ]])
else()
  install(CODE [[
    execute_process(COMMAND "${CMAKE_COMMAND}" -E create_symlink gufo
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gufo-server"
      COMMAND_ERROR_IS_FATAL ANY)
  ]])
endif()
