#include <torch/extension.h>
#include <feat_to_v.h>
#include <feat_to_e.h>


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("feat_to_v_attr", &ext::feat_to_v_attr);
    m.def("feat_to_instance_v", &ext::feat_to_instance_v);
    m.def("feat_to_e", &ext::feat_to_e);
    m.def("feat_to_instance_e", &ext::feat_to_instance_e);
}
