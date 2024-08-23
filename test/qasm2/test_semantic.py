import unittest

from snuqs.qasm2.compiler import Compiler


class SemanticTest(unittest.TestCase):
    #######################
    #### version check ####
    #######################
    def test_valid_version(self):
        file_name = "data/semantic/valid_version.qasm"
        Compiler().compile(file_name)

    def test_invalid_version(self):
        file_name = "data/semantic/invalid_version.qasm"
        self.assertRaises(
            ValueError, lambda: Compiler().compile(file_name))

    ####################
    #### qreg check ####
    ####################
    def test_invalid_qreg_zero_dim(self):
        file_name = "data/semantic/invalid_qreg_zero_dim.qasm"
        self.assertRaises(
            ValueError, lambda: Compiler().compile(file_name))

    def test_invalid_qreg_already_defined(self):
        file_name = "data/semantic/invalid_qreg_already_defined.qasm"
        self.assertRaises(
            LookupError, lambda: Compiler().compile(file_name))

    ####################
    #### creg check ####
    ####################
    def test_invalid_creg_zero_dim(self):
        file_name = "data/semantic/invalid_creg_zero_dim.qasm"
        self.assertRaises(
            ValueError, lambda: Compiler().compile(file_name))

    def test_invalid_creg_already_defined(self):
        file_name = "data/semantic/invalid_creg_already_defined.qasm"
        self.assertRaises( LookupError, lambda: Compiler().compile(file_name))

    ###########################
    #### opaque gate check ####
    ###########################
    def test_invalid_opaque_not_supported(self):
        file_name = "data/semantic/invalid_opaque_not_supported.qasm"
        self.assertRaises(
            ValueError, lambda: Compiler().compile(file_name))

    def test_invalid_opaque_already_defined(self):
        file_name = "data/semantic/invalid_opaque_already_defined.qasm"
        self.assertRaises(
            LookupError, lambda: Compiler().compile(file_name))

    def test_invalid_opaque_dup_ids(self):
        file_name = "data/semantic/invalid_opaque_dup_ids.qasm"
        self.assertRaises(
            LookupError, lambda: Compiler().compile(file_name))

    def test_invalid_opaque_dup_params(self):
        file_name = "data/semantic/invalid_opaque_dup_params.qasm"
        self.assertRaises(
            LookupError, lambda: Compiler().compile(file_name))

    ###########################
    #### gate check ####
    ###########################
    def test_invalid_gate_already_defined(self):
        file_name = "data/semantic/invalid_gate_already_defined.qasm"
        self.assertRaises(LookupError, lambda: Compiler().compile(file_name))

    def test_invalid_gate_dup_ids(self):
        file_name = "data/semantic/invalid_gate_dup_ids.qasm"
        self.assertRaises(
            LookupError, lambda: Compiler().compile(file_name))

    def test_invalid_gate_dup_params(self):
        file_name = "data/semantic/invalid_gate_dup_params.qasm"
        self.assertRaises(
            LookupError, lambda: Compiler().compile(file_name))

#    def test_invalid_gop_u_id_undefined(self):
#        file_name = "data/semantic/invalid_gop_u_id_undefined.qasm"
#        self.assertRaises(
#            LookupError, lambda: Compiler().compile(file_name))
#
#    def test_invalid_gop_u_param_undefined(self):
#        file_name = "data/semantic/invalid_gop_u_param_undefined.qasm"
#        self.assertRaises(
#            LookupError, lambda: Compiler().compile(file_name))
#
#    def test_invalid_gop_u_num_params(self):
#        file_name = "data/semantic/invalid_gop_u_num_params.qasm"
#        self.assertRaises(
#            LookupError, lambda: Compiler().compile(file_name))
#
#    def test_invalid_gop_cx_id_undefined(self):
#        file_name = "data/semantic/invalid_gop_cx_id_undefined.qasm"
#        self.assertRaises(
#            LookupError, lambda: Compiler().compile(file_name))
#
#    def test_invalid_gop_cx_dup_ids(self):
#        file_name = "data/semantic/invalid_gop_cx_dup_ids.qasm"
#        self.assertRaises(
#            ValueError, lambda: Compiler().compile(file_name))
#
#    def test_invalid_gop_barrier_id_undefined(self):
#        file_name = "data/semantic/invalid_barrierop_invalid_target_ref.qasm"
#        self.assertRaises(
#            LookupError, lambda: Compiler().compile(file_name))
#
#    def test_invalid_gop_barrier_dup_ids(self):
#        file_name = "data/semantic/invalid_barrierop_dup_target.qasm"
#        self.assertRaises(
#            ValueError, lambda: Compiler().compile(file_name))
#
#    def test_invalid_gop_custom_gate_undefined(self):
#        file_name = "data/semantic/invalid_gop_custom_gate_undefined.qasm"
#        self.assertRaises(
#            LookupError, lambda: Compiler().compile(file_name))
#
#    def test_invalid_gop_custom_num_ids(self):
#        file_name = "data/semantic/invalid_gop_custom_num_ids.qasm"
#        self.assertRaises(
#            ValueError, lambda: Compiler().compile(file_name))
#
#    def test_invalid_gop_custom_dup_ids(self):
#        file_name = "data/semantic/invalid_gop_custom_dup_ids.qasm"
#        self.assertRaises(
#            ValueError, lambda: Compiler().compile(file_name))
#
#    def test_invalid_gop_custom_num_params(self):
#        file_name = "data/semantic/invalid_gop_custom_num_params.qasm"
#        self.assertRaises(
#            ValueError, lambda: Compiler().compile(file_name))
#
#    def test_invalid_qop_u_qreg_undefined(self):
#        file_name = "data/semantic/invalid_qop_u_qreg_undefined.qasm"
#        self.assertRaises(
#            LookupError, lambda: Compiler().compile(file_name))
#
#    def test_invalid_qop_u_qreg_index(self):
#        file_name = "data/semantic/invalid_qop_u_qreg_index.qasm"
#        self.assertRaises(
#            IndexError, lambda: Compiler().compile(file_name))
#
#    def test_invalid_qop_u_creg_undefined(self):
#        file_name = "data/semantic/invalid_qop_u_creg_undefined.qasm"
#        self.assertRaises(
#            LookupError, lambda: Compiler().compile(file_name))
#
#    def test_invalid_qop_u_num_params(self):
#        file_name = "data/semantic/invalid_qop_u_num_params.qasm"
#        self.assertRaises(
#            ValueError, lambda: Compiler().compile(file_name))
#
#    def test_invalid_qop_cx_qreg_undefined(self):
#        file_name = "data/semantic/invalid_qop_cx_qreg_undefined.qasm"
#        self.assertRaises(
#            LookupError, lambda: Compiler().compile(file_name))
#
#    def test_invalid_qop_cx_qreg_index(self):
#        file_name = "data/semantic/invalid_qop_cx_qreg_index.qasm"
#        self.assertRaises(
#            IndexError, lambda: Compiler().compile(file_name))
#
#    def test_invalid_qop_cx_dup_qregs(self):
#        file_name = "data/semantic/invalid_qop_cx_dup_qregs.qasm"
#        self.assertRaises(
#            ValueError, lambda: Compiler().compile(file_name))
#
#    def test_invalid_qop_measure_qreg_undefined(self):
#        file_name = "data/semantic/invalid_qop_measure_qreg_undefined.qasm"
#        self.assertRaises(
#            LookupError, lambda: Compiler().compile(file_name))
#
#    def test_invalid_qop_measure_qreg_index(self):
#        file_name = "data/semantic/invalid_qop_measure_qreg_index.qasm"
#        self.assertRaises(
#            IndexError, lambda: Compiler().compile(file_name))
#
#    def test_invalid_qop_measure_creg_undefined(self):
#        file_name = "data/semantic/invalid_qop_measure_creg_undefined.qasm"
#        self.assertRaises(
#            LookupError, lambda: Compiler().compile(file_name))
#
#    def test_invalid_qop_measure_creg_index(self):
#        file_name = "data/semantic/invalid_qop_measure_creg_index.qasm"
#        self.assertRaises(
#            IndexError, lambda: Compiler().compile(file_name))
#
#    def test_invalid_qop_measure_dim_mismatch(self):
#        file_name = "data/semantic/invalid_qop_measure_dim_mismatch.qasm"
#        self.assertRaises(
#            IndexError, lambda: Compiler().compile(file_name))
#
#    def test_invalid_qop_reset_qreg_undefined(self):
#        file_name = "data/semantic/invalid_qop_reset_qreg_undefined.qasm"
#        self.assertRaises(
#            LookupError, lambda: Compiler().compile(file_name))
#
#    def test_invalid_qop_reset_qreg_index(self):
#        file_name = "data/semantic/invalid_qop_reset_qreg_index.qasm"
#        self.assertRaises(
#            IndexError, lambda: Compiler().compile(file_name))
#
#    def test_invalid_qop_custom_qreg_undefined(self):
#        file_name = "data/semantic/invalid_qop_custom_qreg_undefined.qasm"
#        self.assertRaises(
#            LookupError, lambda: Compiler().compile(file_name))
#
#    def test_invalid_qop_custom_qreg_index(self):
#        file_name = "data/semantic/invalid_qop_custom_qreg_index.qasm"
#        self.assertRaises(
#            IndexError, lambda: Compiler().compile(file_name))
#
#    def test_invalid_qop_custom_creg_undefined(self):
#        file_name = "data/semantic/invalid_qop_custom_creg_undefined.qasm"
#        self.assertRaises(
#            LookupError, lambda: Compiler().compile(file_name))
#
#    def test_invalid_qop_custom_gate_undefined(self):
#        file_name = "data/semantic/invalid_qop_custom_gate_undefined.qasm"
#        self.assertRaises(
#            LookupError, lambda: Compiler().compile(file_name))
#
#    def test_invalid_qop_custom_num_qregs(self):
#        file_name = "data/semantic/invalid_qop_custom_num_qregs.qasm"
#        self.assertRaises(
#            ValueError, lambda: Compiler().compile(file_name))
#
#    def test_invalid_qop_custom_dup_qregs(self):
#        file_name = "data/semantic/invalid_qop_custom_dup_qregs.qasm"
#        self.assertRaises(
#            ValueError, lambda: Compiler().compile(file_name))
#
#    def test_invalid_qop_custom_num_params(self):
#        file_name = "data/semantic/invalid_qop_custom_num_params.qasm"
#        self.assertRaises(
#            ValueError, lambda: Compiler().compile(file_name))

    ############################
    #### if statement check ####
    ############################
    def test_invalid_if_creg_undefined(self):
        file_name = "data/semantic/invalid_if_creg_undefined.qasm"
        self.assertRaises(
            LookupError, lambda: Compiler().compile(file_name))

    def test_invalid_if_not_creg_ref(self):
        file_name = "data/semantic/invalid_if_not_creg_ref.qasm"
        self.assertRaises(
            TypeError, lambda: Compiler().compile(file_name))

    #######################
    #### barrier check ####
    #######################
    def test_invalid_barrier_arg_undefined(self):
        file_name = "data/semantic/invalid_barrier_arg_undefined.qasm"
        self.assertRaises(
            LookupError, lambda: Compiler().compile(file_name))

    def test_invalid_barrier_duplicated_args_1(self):
        file_name = "data/semantic/invalid_barrier_duplicated_args_1.qasm"
        self.assertRaises(
            ValueError, lambda: Compiler().compile(file_name))

    def test_invalid_barrier_duplicated_args_2(self):
        file_name = "data/semantic/invalid_barrier_duplicated_args_2.qasm"
        self.assertRaises(
            ValueError, lambda: Compiler().compile(file_name))

    def test_invalid_barrier_illegal_qreg_index(self):
        file_name = "data/semantic/invalid_barrier_illegal_qreg_index.qasm"
        self.assertRaises(
            IndexError, lambda: Compiler().compile(file_name))


if __name__ == '__main__':
    unittest.main()
