import unittest

from snuqs.compiler.qasm_compiler import QasmCompiler


class QasmSemanticTest(unittest.TestCase):
    def test_zero_dim_qreg(self):
        with open("qasm/compiler/zero_dim_qreg.qasm") as f:
            qasm = f.read()
        comp = QasmCompiler()
        self.assertRaises(ValueError, lambda: comp.compile(qasm))

    def test_zero_dim_creg(self):
        with open("qasm/compiler/zero_dim_creg.qasm") as f:
            qasm = f.read()
        comp = QasmCompiler()
        self.assertRaises(ValueError, lambda: comp.compile(qasm))

    def test_dup_qreg_name(self):
        with open("qasm/compiler/dup_qreg_name.qasm") as f:
            qasm = f.read()
        comp = QasmCompiler()
        self.assertRaises(LookupError, lambda: comp.compile(qasm))

    def test_dup_creg_name(self):
        with open("qasm/compiler/dup_creg_name.qasm") as f:
            qasm = f.read()
        comp = QasmCompiler()
        self.assertRaises(LookupError, lambda: comp.compile(qasm))

    def test_opaque_dup_idlist(self):
        with open("qasm/compiler/opaque_dup_idlist.qasm") as f:
            qasm = f.read()
        comp = QasmCompiler()
        self.assertRaises(LookupError, lambda: comp.compile(qasm))

    def test_opaque_dup_paramlist(self):
        with open("qasm/compiler/opaque_dup_paramlist.qasm") as f:
            qasm = f.read()
        comp = QasmCompiler()
        self.assertRaises(LookupError, lambda: comp.compile(qasm))

    def test_gatedecl_dup_idlist(self):
        with open("qasm/compiler/gatedecl_dup_idlist.qasm") as f:
            qasm = f.read()
        comp = QasmCompiler()
        self.assertRaises(LookupError, lambda: comp.compile(qasm))

    def test_gatedecl_dup_paramlist(self):
        with open("qasm/compiler/gatedecl_dup_paramlist.qasm") as f:
            qasm = f.read()
        comp = QasmCompiler()
        self.assertRaises(LookupError, lambda: comp.compile(qasm))

    def test_if_undefined(self):
        with open("qasm/compiler/if_undefined_creg.qasm") as f:
            qasm = f.read()
        comp = QasmCompiler()
        self.assertRaises(LookupError, lambda: comp.compile(qasm))

    def test_barrier_invalid_qreg_ref(self):
        with open("qasm/compiler/barrier_invalid_qreg_ref.qasm") as f:
            qasm = f.read()
        comp = QasmCompiler()
        self.assertRaises(LookupError, lambda: comp.compile(qasm))

    def test_barrier_invalid_qreg_indexing(self):
        with open("qasm/compiler/barrier_invalid_qreg_indexing.qasm") as f:
            qasm = f.read()
        comp = QasmCompiler()
        self.assertRaises(IndexError, lambda: comp.compile(qasm))

    def test_uop_invalid_target_ref(self):
        with open("qasm/compiler/uop_invalid_target_ref.qasm") as f:
            qasm = f.read()
        comp = QasmCompiler()
        self.assertRaises(LookupError, lambda: comp.compile(qasm))

    def test_uop_invalid_num_exprs(self):
        with open("qasm/compiler/uop_invalid_num_exprs.qasm") as f:
            qasm = f.read()
        comp = QasmCompiler()
        self.assertRaises(ValueError, lambda: comp.compile(qasm))

    def test_uop_invalid_expr(self):
        with open("qasm/compiler/uop_invalid_expr.qasm") as f:
            qasm = f.read()
        comp = QasmCompiler()
        self.assertRaises(LookupError, lambda: comp.compile(qasm))

    def test_cxop_invalid_target_ref(self):
        with open("qasm/compiler/cxop_invalid_target_ref.qasm") as f:
            qasm = f.read()
        comp = QasmCompiler()
        self.assertRaises(LookupError, lambda: comp.compile(qasm))

    def test_cxop_dup_target(self):
        with open("qasm/compiler/cxop_dup_target.qasm") as f:
            qasm = f.read()
        comp = QasmCompiler()
        self.assertRaises(ValueError, lambda: comp.compile(qasm))

    def test_barrierop_invalid_target_ref(self):
        with open("qasm/compiler/barrierop_invalid_target_ref.qasm") as f:
            qasm = f.read()
        comp = QasmCompiler()
        self.assertRaises(LookupError, lambda: comp.compile(qasm))

    def test_barrierop_dup_target(self):
        with open("qasm/compiler/barrierop_dup_target.qasm") as f:
            qasm = f.read()
        comp = QasmCompiler()
        self.assertRaises(ValueError, lambda: comp.compile(qasm))

    def test_customop_invalid_gate_ref(self):
        with open("qasm/compiler/customop_invalid_gate_ref.qasm") as f:
            qasm = f.read()
        comp = QasmCompiler()
        self.assertRaises(LookupError, lambda: comp.compile(qasm))

    def test_customop_invalid_num_args(self):
        with open("qasm/compiler/customop_invalid_num_args.qasm") as f:
            qasm = f.read()
        comp = QasmCompiler()
        self.assertRaises(ValueError, lambda: comp.compile(qasm))

    def test_customop_dup_targets(self):
        with open("qasm/compiler/customop_dup_targets.qasm") as f:
            qasm = f.read()
        comp = QasmCompiler()
        self.assertRaises(ValueError, lambda: comp.compile(qasm))

    def test_customop_invalid_num_exprs(self):
        with open("qasm/compiler/customop_invalid_num_exprs.qasm") as f:
            qasm = f.read()
        comp = QasmCompiler()
        self.assertRaises(ValueError, lambda: comp.compile(qasm))

    def test_qop_u_invalid_num_exprs(self):
        with open("qasm/compiler/qop_u_invalid_num_exprs.qasm") as f:
            qasm = f.read()
        comp = QasmCompiler()
        self.assertRaises(ValueError, lambda: comp.compile(qasm))

    def test_qop_u_invalid_creg_ref(self):
        with open("qasm/compiler/qop_u_invalid_creg_ref.qasm") as f:
            qasm = f.read()
        comp = QasmCompiler()
        self.assertRaises(LookupError, lambda: comp.compile(qasm))

    def test_qop_custom_invalid_gate_ref(self):
        with open("qasm/compiler/qop_custom_invalid_gate_ref.qasm") as f:
            qasm = f.read()
        comp = QasmCompiler()
        self.assertRaises(LookupError, lambda: comp.compile(qasm))

    def test_qop_custom_invalid_qreg_ref(self):
        with open("qasm/compiler/qop_custom_invalid_qreg_ref.qasm") as f:
            qasm = f.read()
        comp = QasmCompiler()
        self.assertRaises(LookupError, lambda: comp.compile(qasm))
    
    def test_qop_custom_invalid_creg_ref(self):
        with open("qasm/compiler/qop_custom_invalid_creg_ref.qasm") as f:
            qasm = f.read()
        comp = QasmCompiler()
        self.assertRaises(LookupError, lambda: comp.compile(qasm))

    def test_qop_custom_dup_targets(self):
        with open("qasm/compiler/qop_custom_dup_targets.qasm") as f:
            qasm = f.read()
        comp = QasmCompiler()
        self.assertRaises(ValueError, lambda: comp.compile(qasm))

    def test_qop_custom_invalid_num_args(self):
        with open("qasm/compiler/qop_custom_invalid_num_args.qasm") as f:
            qasm = f.read()
        comp = QasmCompiler()
        self.assertRaises(ValueError, lambda: comp.compile(qasm))

    def test_qop_custom_invalid_num_exprs(self):
        with open("qasm/compiler/qop_custom_invalid_num_exprs.qasm") as f:
            qasm = f.read()
        comp = QasmCompiler()
        self.assertRaises(ValueError, lambda: comp.compile(qasm))

    def test_qop_barrier_dup_targets(self):
        with open("qasm/compiler/barrier_dup_targets.qasm") as f:
            qasm = f.read()
        comp = QasmCompiler()
        self.assertRaises(ValueError, lambda: comp.compile(qasm))


if __name__ == '__main__':
    unittest.main()
