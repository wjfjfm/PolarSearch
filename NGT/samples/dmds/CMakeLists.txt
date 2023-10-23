if( ${UNIX} )
	include_directories("${PROJECT_BINARY_DIR}/lib")
        include_directories("${PROJECT_SOURCE_DIR}/lib")
        link_directories("${PROJECT_SOURCE_DIR}/lib/NGT")

	add_executable(dmds_test dmds_test.cpp)
	add_dependencies(dmds_test ngt)
	target_link_libraries(dmds_test ngt pthread)
endif()
