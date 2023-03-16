file(REMOVE_RECURSE
  "libnano_grad_backend_shared.pdb"
  "libnano_grad_backend_shared.so"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/nano_grad_backend_shared.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
