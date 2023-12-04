import unittest

from snuqs.compiler.qasm_compiler import QasmCompiler


class QasmSemanticTest(unittest.TestCase):
    def test_invalid_zero_dim_qreg(self):
        file_name = "qasm/semantic/invalid_zero_dim_qreg.qasm"
        self.assertRaises(ValueError, lambda: QasmCompiler().compile(file_name))

    def test_invalid_zero_dim_creg(self):
        file_name = "qasm/semantic/invalid_zero_dim_creg.qasm"
        self.assertRaises(ValueError, lambda: QasmCompiler().compile(file_name))

    def test_invalid_dup_qreg_name(self):
        file_name = "qasm/semantic/invalid_dup_qreg_name.qasm"
        self.assertRaises(LookupError, lambda: QasmCompiler().compile(file_name))

    def test_invalid_dup_creg_name(self):
        file_name = "qasm/semantic/invalid_dup_creg_name.qasm"
        self.assertRaises(LookupError, lambda: QasmCompiler().compile(file_name))

    def test_invalid_opaque_dup_idlist(self):
        file_name = "qasm/semantic/invalid_opaque_dup_idlist.qasm"
        self.assertRaises(LookupError, lambda: QasmCompiler().compile(file_name))

    def test_invalid_opaque_dup_paramlist(self):
        file_name = "qasm/semantic/invalid_opaque_dup_paramlist.qasm"
        self.assertRaises(LookupError, lambda: QasmCompiler().compile(file_name))

    def test_invalid_gatedecl_dup_idlist(self):
        file_name = "qasm/semantic/invalid_gatedecl_dup_idlist.qasm"
        self.assertRaises(LookupError, lambda: QasmCompiler().compile(file_name))

    def test_invalid_gatedecl_dup_paramlist(self):
        file_name = "qasm/semantic/invalid_gatedecl_dup_paramlist.qasm"
        self.assertRaises(LookupError, lambda: QasmCompiler().compile(file_name))

    def test_invalid_if_undefined(self):
        file_name = "qasm/semantic/invalid_if_undefined_creg.qasm"
        self.assertRaises(LookupError, lambda: QasmCompiler().compile(file_name))

    def test_invalid_barrier_invalid_qreg_ref(self):
        file_name = "qasm/semantic/invalid_barrier_invalid_qreg_ref.qasm"
        self.assertRaises(LookupError, lambda: QasmCompiler().compile(file_name))

    def test_invalid_barrier_invalid_qreg_indexing(self):
        file_name = "qasm/semantic/invalid_barrier_invalid_qreg_indexing.qasm"
        self.assertRaises(IndexError, lambda: QasmCompiler().compile(file_name))

    def test_invalid_uop_invalid_target_ref(self):
        file_name = "qasm/semantic/invalid_uop_invalid_target_ref.qasm"
        self.assertRaises(LookupError, lambda: QasmCompiler().compile(file_name))

    def test_invalid_uop_invalid_num_exprs(self):
        file_name = "qasm/semantic/invalid_uop_invalid_num_exprs.qasm"
        self.assertRaises(ValueError, lambda: QasmCompiler().compile(file_name))

    def test_invalid_uop_invalid_expr(self):
        file_name = "qasm/semantic/invalid_uop_invalid_expr.qasm"
        self.assertRaises(LookupError, lambda: QasmCompiler().compile(file_name))

    def test_invalid_cxop_invalid_target_ref(self):
        file_name = "qasm/semantic/invalid_cxop_invalid_target_ref.qasm"
        self.assertRaises(LookupError, lambda: QasmCompiler().compile(file_name))

    def test_invalid_cxop_dup_target(self):
        file_name = "qasm/semantic/invalid_cxop_dup_target.qasm"
        self.assertRaises(ValueError, lambda: QasmCompiler().compile(file_name))

    def test_invalid_barrierop_invalid_target_ref(self):
        file_name = "qasm/semantic/invalid_barrierop_invalid_target_ref.qasm"
        self.assertRaises(LookupError, lambda: QasmCompiler().compile(file_name))

    def test_invalid_barrierop_dup_target(self):
        file_name = "qasm/semantic/invalid_barrierop_dup_target.qasm"
        self.assertRaises(ValueError, lambda: QasmCompiler().compile(file_name))

    def test_invalid_customop_invalid_gate_ref(self):
        file_name = "qasm/semantic/invalid_customop_invalid_gate_ref.qasm"
        self.assertRaises(LookupError, lambda: QasmCompiler().compile(file_name))

    def test_invalid_customop_invalid_num_args(self):
        file_name = "qasm/semantic/invalid_customop_invalid_num_args.qasm"
        self.assertRaises(ValueError, lambda: QasmCompiler().compile(file_name))

    def test_invalid_customop_dup_targets(self):
        file_name = "qasm/semantic/invalid_customop_dup_targets.qasm"
        self.assertRaises(ValueError, lambda: QasmCompiler().compile(file_name))

    def test_invalid_customop_invalid_num_exprs(self):
        file_name = "qasm/semantic/invalid_customop_invalid_num_exprs.qasm"
        self.assertRaises(ValueError, lambda: QasmCompiler().compile(file_name))

    def test_invalid_qop_u_invalid_num_exprs(self):
        file_name = "qasm/semantic/invalid_qop_u_invalid_num_exprs.qasm"
        self.assertRaises(ValueError, lambda: QasmCompiler().compile(file_name))

    def test_invalid_qop_u_invalid_creg_ref(self):
        file_name = "qasm/semantic/invalid_qop_u_invalid_creg_ref.qasm"
        self.assertRaises(LookupError, lambda: QasmCompiler().compile(file_name))

    def test_invalid_qop_custom_invalid_gate_ref(self):
        file_name = "qasm/semantic/invalid_qop_custom_invalid_gate_ref.qasm"
        self.assertRaises(LookupError, lambda: QasmCompiler().compile(file_name))

    def test_invalid_qop_custom_invalid_qreg_ref(self):
        file_name = "qasm/semantic/invalid_qop_custom_invalid_qreg_ref.qasm"
        self.assertRaises(LookupError, lambda: QasmCompiler().compile(file_name))
    
    def test_invalid_qop_custom_invalid_creg_ref(self):
        file_name = "qasm/semantic/invalid_qop_custom_invalid_creg_ref.qasm"
        self.assertRaises(LookupError, lambda: QasmCompiler().compile(file_name))

    def test_invalid_qop_custom_dup_targets(self):
        file_name = "qasm/semantic/invalid_qop_custom_dup_targets.qasm"
        self.assertRaises(ValueError, lambda: QasmCompiler().compile(file_name))

    def test_invalid_qop_custom_invalid_num_args(self):
        file_name = "qasm/semantic/invalid_qop_custom_invalid_num_args.qasm"
        self.assertRaises(ValueError, lambda: QasmCompiler().compile(file_name))

    def test_invalid_qop_custom_invalid_num_exprs(self):
        file_name = "qasm/semantic/invalid_qop_custom_invalid_num_exprs.qasm"
        self.assertRaises(ValueError, lambda: QasmCompiler().compile(file_name))

    def test_invalid_qop_barrier_dup_targets(self):
        file_name = "qasm/semantic/invalid_barrier_dup_targets.qasm"
        self.assertRaises(ValueError, lambda: QasmCompiler().compile(file_name))


if __name__ == '__main__':
    unittest.main()
