import unittest

from snuqs.compiler.qasm_compiler import QasmCompiler


class QasmSemanticTest(unittest.TestCase):
    def test_invalid_qreg_zero_dim(self):
        file_name = "qasm/semantic/invalid_qreg_zero_dim.qasm"
        self.assertRaises(
            ValueError, lambda: QasmCompiler().compile(file_name))

    def test_invalid_qreg_already_defined(self):
        file_name = "qasm/semantic/invalid_qreg_already_defined.qasm"
        self.assertRaises(
            LookupError, lambda: QasmCompiler().compile(file_name))

    def test_invalid_creg_zero_dim(self):
        file_name = "qasm/semantic/invalid_creg_zero_dim.qasm"
        self.assertRaises(
            ValueError, lambda: QasmCompiler().compile(file_name))

    def test_invalid_creg_already_defined(self):
        file_name = "qasm/semantic/invalid_creg_already_defined.qasm"
        self.assertRaises(
            LookupError, lambda: QasmCompiler().compile(file_name))

    def test_invalid_opaque_already_defined(self):
        file_name = "qasm/semantic/invalid_opaque_already_defined.qasm"
        self.assertRaises(
            LookupError, lambda: QasmCompiler().compile(file_name))

    def test_invalid_opaque_dup_ids(self):
        file_name = "qasm/semantic/invalid_opaque_dup_ids.qasm"
        self.assertRaises(
            LookupError, lambda: QasmCompiler().compile(file_name))

    def test_invalid_opaque_dup_params(self):
        file_name = "qasm/semantic/invalid_opaque_dup_params.qasm"
        self.assertRaises(
            LookupError, lambda: QasmCompiler().compile(file_name))

    def test_invalid_gate_already_defined(self):
        file_name = "qasm/semantic/invalid_gate_already_defined.qasm"
        self.assertRaises(
            LookupError, lambda: QasmCompiler().compile(file_name))

    def test_invalid_gate_dup_ids(self):
        file_name = "qasm/semantic/invalid_gate_dup_ids.qasm"
        self.assertRaises(
            LookupError, lambda: QasmCompiler().compile(file_name))

    def test_invalid_gate_dup_params(self):
        file_name = "qasm/semantic/invalid_gate_dup_params.qasm"
        self.assertRaises(
            LookupError, lambda: QasmCompiler().compile(file_name))

    def test_invalid_gop_u_id_undefined(self):
        file_name = "qasm/semantic/invalid_gop_u_id_undefined.qasm"
        self.assertRaises(
            LookupError, lambda: QasmCompiler().compile(file_name))

    def test_invalid_gop_u_param_undefined(self):
        file_name = "qasm/semantic/invalid_gop_u_param_undefined.qasm"
        self.assertRaises(
            LookupError, lambda: QasmCompiler().compile(file_name))

    def test_invalid_gop_u_num_params(self):
        file_name = "qasm/semantic/invalid_gop_u_num_params.qasm"
        self.assertRaises(
            LookupError, lambda: QasmCompiler().compile(file_name))

    def test_invalid_gop_cx_id_undefined(self):
        file_name = "qasm/semantic/invalid_gop_cx_id_undefined.qasm"
        self.assertRaises(
            LookupError, lambda: QasmCompiler().compile(file_name))

    def test_invalid_gop_cx_dup_ids(self):
        file_name = "qasm/semantic/invalid_gop_cx_dup_ids.qasm"
        self.assertRaises(
            ValueError, lambda: QasmCompiler().compile(file_name))

    def test_invalid_gop_barrier_id_undefined(self):
        file_name = "qasm/semantic/invalid_barrierop_invalid_target_ref.qasm"
        self.assertRaises(
            LookupError, lambda: QasmCompiler().compile(file_name))

    def test_invalid_gop_barrier_dup_ids(self):
        file_name = "qasm/semantic/invalid_barrierop_dup_target.qasm"
        self.assertRaises(
            ValueError, lambda: QasmCompiler().compile(file_name))

    def test_invalid_gop_custom_gate_undefined(self):
        file_name = "qasm/semantic/invalid_gop_custom_gate_undefined.qasm"
        self.assertRaises(
            LookupError, lambda: QasmCompiler().compile(file_name))

    def test_invalid_gop_custom_num_ids(self):
        file_name = "qasm/semantic/invalid_gop_custom_num_ids.qasm"
        self.assertRaises(
            ValueError, lambda: QasmCompiler().compile(file_name))

    def test_invalid_gop_custom_dup_ids(self):
        file_name = "qasm/semantic/invalid_gop_custom_dup_ids.qasm"
        self.assertRaises(
            ValueError, lambda: QasmCompiler().compile(file_name))

    def test_invalid_gop_custom_num_params(self):
        file_name = "qasm/semantic/invalid_gop_custom_num_params.qasm"
        self.assertRaises(
            ValueError, lambda: QasmCompiler().compile(file_name))

    def test_invalid_qop_u_qreg_undefined(self):
        file_name = "qasm/semantic/invalid_qop_u_qreg_undefined.qasm"
        self.assertRaises(
            LookupError, lambda: QasmCompiler().compile(file_name))

    def test_invalid_qop_u_qreg_index(self):
        file_name = "qasm/semantic/invalid_qop_u_qreg_index.qasm"
        self.assertRaises(
            IndexError, lambda: QasmCompiler().compile(file_name))

    def test_invalid_qop_u_creg_undefined(self):
        file_name = "qasm/semantic/invalid_qop_u_creg_undefined.qasm"
        self.assertRaises(
            LookupError, lambda: QasmCompiler().compile(file_name))

    def test_invalid_qop_u_num_params(self):
        file_name = "qasm/semantic/invalid_qop_u_num_params.qasm"
        self.assertRaises(
            ValueError, lambda: QasmCompiler().compile(file_name))

    def test_invalid_qop_cx_qreg_undefined(self):
        file_name = "qasm/semantic/invalid_qop_cx_qreg_undefined.qasm"
        self.assertRaises(
            LookupError, lambda: QasmCompiler().compile(file_name))

    def test_invalid_qop_cx_qreg_index(self):
        file_name = "qasm/semantic/invalid_qop_cx_qreg_index.qasm"
        self.assertRaises(
            IndexError, lambda: QasmCompiler().compile(file_name))

    def test_invalid_qop_cx_dup_qregs(self):
        file_name = "qasm/semantic/invalid_qop_cx_dup_qregs.qasm"
        self.assertRaises(
            ValueError, lambda: QasmCompiler().compile(file_name))

    def test_invalid_qop_measure_qreg_undefined(self):
        file_name = "qasm/semantic/invalid_qop_measure_qreg_undefined.qasm"
        self.assertRaises(
            LookupError, lambda: QasmCompiler().compile(file_name))

    def test_invalid_qop_measure_qreg_index(self):
        file_name = "qasm/semantic/invalid_qop_measure_qreg_index.qasm"
        self.assertRaises(
            IndexError, lambda: QasmCompiler().compile(file_name))

    def test_invalid_qop_measure_creg_undefined(self):
        file_name = "qasm/semantic/invalid_qop_measure_creg_undefined.qasm"
        self.assertRaises(
            LookupError, lambda: QasmCompiler().compile(file_name))

    def test_invalid_qop_measure_creg_index(self):
        file_name = "qasm/semantic/invalid_qop_measure_creg_index.qasm"
        self.assertRaises(
            IndexError, lambda: QasmCompiler().compile(file_name))

    def test_invalid_qop_measure_dim_mismatch(self):
        file_name = "qasm/semantic/invalid_qop_measure_dim_mismatch.qasm"
        self.assertRaises(
            IndexError, lambda: QasmCompiler().compile(file_name))

    def test_invalid_qop_reset_qreg_undefined(self):
        file_name = "qasm/semantic/invalid_qop_reset_qreg_undefined.qasm"
        self.assertRaises(
            LookupError, lambda: QasmCompiler().compile(file_name))

    def test_invalid_qop_reset_qreg_index(self):
        file_name = "qasm/semantic/invalid_qop_reset_qreg_index.qasm"
        self.assertRaises(
            IndexError, lambda: QasmCompiler().compile(file_name))

    def test_invalid_qop_custom_qreg_undefined(self):
        file_name = "qasm/semantic/invalid_qop_custom_qreg_undefined.qasm"
        self.assertRaises(
            LookupError, lambda: QasmCompiler().compile(file_name))

    def test_invalid_qop_custom_qreg_index(self):
        file_name = "qasm/semantic/invalid_qop_custom_qreg_index.qasm"
        self.assertRaises(
            IndexError, lambda: QasmCompiler().compile(file_name))

    def test_invalid_qop_custom_creg_undefined(self):
        file_name = "qasm/semantic/invalid_qop_custom_creg_undefined.qasm"
        self.assertRaises(
            LookupError, lambda: QasmCompiler().compile(file_name))

    def test_invalid_qop_custom_gate_undefined(self):
        file_name = "qasm/semantic/invalid_qop_custom_gate_undefined.qasm"
        self.assertRaises(
            LookupError, lambda: QasmCompiler().compile(file_name))

    def test_invalid_qop_custom_num_qregs(self):
        file_name = "qasm/semantic/invalid_qop_custom_num_qregs.qasm"
        self.assertRaises(
            ValueError, lambda: QasmCompiler().compile(file_name))

    def test_invalid_qop_custom_dup_qregs(self):
        file_name = "qasm/semantic/invalid_qop_custom_dup_qregs.qasm"
        self.assertRaises(
            ValueError, lambda: QasmCompiler().compile(file_name))

    def test_invalid_qop_custom_num_params(self):
        file_name = "qasm/semantic/invalid_qop_custom_num_params.qasm"
        self.assertRaises(
            ValueError, lambda: QasmCompiler().compile(file_name))

    def test_invalid_if_creg_undefined(self):
        file_name = "qasm/semantic/invalid_if_creg_undefined.qasm"
        self.assertRaises(
            LookupError, lambda: QasmCompiler().compile(file_name))

    def test_invalid_barrier_qreg_undefined(self):
        file_name = "qasm/semantic/invalid_barrier_qreg_undefined.qasm"
        self.assertRaises(
            LookupError, lambda: QasmCompiler().compile(file_name))

    def test_invalid_barrier_dup_qregs(self):
        file_name = "qasm/semantic/invalid_barrier_dup_qregs.qasm"
        self.assertRaises(
            ValueError, lambda: QasmCompiler().compile(file_name))

    def test_invalid_barrier_qreg_index(self):
        file_name = "qasm/semantic/invalid_barrier_qreg_index.qasm"
        self.assertRaises(
            IndexError, lambda: QasmCompiler().compile(file_name))


if __name__ == '__main__':
    unittest.main()
