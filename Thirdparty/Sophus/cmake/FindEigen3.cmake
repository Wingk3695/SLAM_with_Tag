# Search for the Eigen3 header files
find_path(EIGEN3_INCLUDE_DIR Eigen/Core PATHS /usr/local/include/eigen3 /usr/include/eigen3)

# Handle errors if Eigen is not found
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Eigen3 DEFAULT_MSG EIGEN3_INCLUDE_DIR)

# Mark as advanced to hide it in CMake GUI
mark_as_advanced(EIGEN3_INCLUDE_DIR)