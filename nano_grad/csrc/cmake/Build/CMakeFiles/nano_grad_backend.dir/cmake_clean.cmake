file(REMOVE_RECURSE
  "nano_grad_backend"
  "nano_grad_backend.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/nano_grad_backend.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
