#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <complex>

#include "buffer/buffer.h"
#include "buffer/buffer_cpu.h"
#include "buffer/buffer_cuda.h"
#include "buffer/buffer_pinned.h"
#include "buffer/buffer_storage.h"
#include "core/cuda/runtime.h"
#include "core/runtime.h"
#include "device_types.h"
#include "fs/fs.h"
#include "functionals/functionals.h"
#include "operation/gate_operations.h"
#include "operation/operation.h"
#include "result_types/state_vector.h"

namespace py = pybind11;

#define GATEOP(name)                                                          \
  py::class_<name, GateOperation, std::shared_ptr<name>>(                     \
      m_gate_operations, #name, py::buffer_protocol())                        \
      .def_buffer([](name &g) -> py::buffer_info {                            \
        return py::buffer_info(                                               \
            g.data(), sizeof(std::complex<double>),                           \
            py::format_descriptor<std::complex<double>>::format(), g.dim(),   \
            g.shape(), g.stride());                                           \
      })                                                                      \
      .def(py::init<const std::vector<size_t> &, const std::vector<size_t> &, \
                    size_t>(),                                                \
           py::arg("targets"), py::kw_only(),                                 \
           py::arg("ctrl_modifiers") = std::vector<size_t>(),                 \
           py::arg("power") = 1)

#define GATEOP1(name)                                                       \
  py::class_<name, GateOperation, std::shared_ptr<name>>(                   \
      m_gate_operations, #name, py::buffer_protocol())                      \
      .def_buffer([](name &g) -> py::buffer_info {                          \
        return py::buffer_info(                                             \
            g.data(), sizeof(std::complex<double>),                         \
            py::format_descriptor<std::complex<double>>::format(), g.dim(), \
            g.shape(), g.stride());                                         \
      })                                                                    \
      .def(py::init<const std::vector<size_t> &, double,                    \
                    const std::vector<size_t> &, size_t>(),                 \
           py::arg("targets"), py::arg("angle"), py::kw_only(),             \
           py::arg("ctrl_modifiers") = std::vector<size_t>(),               \
           py::arg("power") = 1)

#define GATEOP3(name)                                                       \
  py::class_<name, GateOperation, std::shared_ptr<name>>(                   \
      m_gate_operations, #name, py::buffer_protocol())                      \
      .def_buffer([](name &g) -> py::buffer_info {                          \
        return py::buffer_info(                                             \
            g.data(), sizeof(std::complex<double>),                         \
            py::format_descriptor<std::complex<double>>::format(), g.dim(), \
            g.shape(), g.stride());                                         \
      })                                                                    \
      .def(py::init<const std::vector<size_t> &, double, double, double,    \
                    const std::vector<size_t> &, size_t>(),                 \
           py::arg("targets"), py::arg("angle_1"), py::arg("angle_2"),      \
           py::arg("angle_3"), py::kw_only(),                               \
           py::arg("ctrl_modifiers") = std::vector<size_t>(),               \
           py::arg("power") = 1)

PYBIND11_MODULE(_C, m) {
  m.doc() = "SnuQS Pybind11 module.";

  // Operation
  auto m_operation = m.def_submodule("operation");
  py::class_<Operation, std::shared_ptr<Operation>>(m_operation, "Operation")
      .def(py::init<const std::vector<size_t> &>())
      .def_property("targets", &Operation::get_targets,
                    &Operation::set_targets);

  // GateOperation
  py::class_<GateOperation, Operation, std::shared_ptr<GateOperation>>(
      m_operation, "GateOperation")
      .def(py::init<GateOperationType, const std::vector<size_t> &,
                    const std::vector<double> &, const std::vector<size_t> &,
                    size_t>())
      .def("sliceable", &GateOperation::sliceable)
      .def("slice", &GateOperation::slice)
      .def("__repr__", &GateOperation::formatted_string);

  auto m_gate_operations = m_operation.def_submodule("gate_operations");
  GATEOP(Identity);
  GATEOP(Hadamard);
  GATEOP(PauliX);
  GATEOP(PauliY);
  GATEOP(PauliZ);
  GATEOP(CX);
  GATEOP(CY);
  GATEOP(CZ);
  GATEOP(S);
  GATEOP(Si);
  GATEOP(T);
  GATEOP(Ti);
  GATEOP(V);
  GATEOP(Vi);
  GATEOP1(PhaseShift);
  GATEOP1(CPhaseShift);
  GATEOP1(CPhaseShift00);
  GATEOP1(CPhaseShift01);
  GATEOP1(CPhaseShift10);
  GATEOP1(RotX);
  GATEOP1(RotY);
  GATEOP1(RotZ);
  GATEOP(Swap);
  GATEOP(ISwap);
  GATEOP1(PSwap);
  GATEOP1(XY);
  GATEOP1(XX);
  GATEOP1(YY);
  GATEOP1(ZZ);
  GATEOP(CCNot);
  GATEOP(CSwap);
  GATEOP3(U);
  GATEOP1(GPhase);

  // Functionals
  auto m_functionals = m.def_submodule("functionals");
  m_functionals.def("apply", &functionals::apply);
  m_functionals.def("initialize_zero", &functionals::initialize_zero);
  m_functionals.def("initialize_basis_z", &functionals::initialize_basis_z);

  // StateVector
  auto m_result_types = m.def_submodule("result_types");
  py::class_<StateVector, std::shared_ptr<StateVector>>(
      m_result_types, "StateVector", py::buffer_protocol())
      .def_buffer([](StateVector &sv) -> py::buffer_info {
        return py::buffer_info(
            sv.ptr(),                     /* Pointer to buffer */
            sizeof(std::complex<double>), /* Size of one scalar */
            py::format_descriptor<std::complex<double>>::format(), /* Python
                                                       struct-style format
                                                       descriptor */
            sv.dim(), sv.shape(), {sizeof(std::complex<double>)});
      })
      .def(py::init<size_t>())
      .def(py::init<DeviceType, size_t>())
      .def("cpu", &StateVector::cpu)
      .def("cuda", &StateVector::cuda)
      .def("slice", &StateVector::slice)
      .def("copy", &StateVector::copy)
      .def("__repr__", &StateVector::formatted_string);

  auto m_core = m.def_submodule("core");
  m_core.def("mem_info", &mem_info)
      .def("mem_info", &mem_info)
      .def("attach_fs", &attach_fs)
      .def("detach_fs", &detach_fs)
      .def("is_attached_fs", &is_attached_fs)
      .def("get_fs", &get_fs);

  auto m_core_cuda = m_core.def_submodule("cuda");
  m_core_cuda.def("mem_info", &cu::mem_info)
      .def("device_count", &cu::device_count)
      .def("get_device", &cu::get_device)
      .def("set_device", &cu::set_device)
      .def("device_synchronize", &cu::device_synchronize);

  // Device
  py::enum_<DeviceType>(m, "DeviceType")
      .value("CPU", DeviceType::CPU)
      .value("CUDA", DeviceType::CUDA)
      .value("STORAGE", DeviceType::STORAGE)
      .export_values();

  // FS
  auto m_fs = m.def_submodule("fs");
  py::class_<fs_addr_t>(m_fs, "fs_addr_t")
      .def("__repr__", &fs_addr_t::formatted_string);
  py::class_<FS, std::shared_ptr<FS>>(m_fs, "FS")
      .def(py::init<size_t, size_t, const std::vector<std::string> &>())
      .def("alloc", &FS::alloc)
      .def("free", &FS::free)
      .def("dump", &FS::dump);

  //  // Buffer
  //  auto m_buffer = m.def_submodule("buffer");
  //  py::class_<Buffer>(m_buffer, "Buffer")
  //      .def("ptr", &Buffer::ptr)
  //      .def("count", &Buffer::count)
  //      .def("__repr__", &Buffer::formatted_string);
  //  py::class_<BufferCPU, Buffer>(m_buffer,
  //  "BufferCPU").def(py::init<size_t>()); py::class_<BufferCUDA,
  //  Buffer>(m_buffer, "BufferCUDA")
  //      .def(py::init<size_t>());
  //  py::class_<BufferStorage, Buffer>(m_buffer, "BufferStorage")
  //      .def(py::init<size_t>());
}
