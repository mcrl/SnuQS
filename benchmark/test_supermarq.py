
from Supermarq.ghz import GHZ
circ_braket = GHZ(3)
circ_braket.run(1000)

circ_snuqs = GHZ(3, backend='snuqs')
circ_snuqs.run(1000)

"""
from Supermarq.mermin_bell import MerminBell

circ_braket = MerminBell(3)
circ_braket.run(1000)

circ_snuqs = MerminBell(3, backend='snuqs')
circ_snuqs.run(1000)

from Supermarq.zz_swap_qaoa import ZZ_Swap_QAOA

circ_braket = ZZ_Swap_QAOA(3)
circ_braket.run(1000)

circ_snuqs = ZZ_Swap_QAOA(3, backend='snuqs')
circ_snuqs.run(1000)


from Supermarq.vanilla_qaoa import Vanilla_QAOA

circ_braket = Vanilla_QAOA(3)
circ_braket.run(1000)

circ_snuqs = Vanilla_QAOA(3, backend='snuqs')
circ_snuqs.run(1000)


from Supermarq.vqe import VQE

circ_braket = VQE(3)
circ_braket.run(1000)

circ_snuqs = VQE(3, backend='snuqs')
circ_snuqs.run(1000)


from Supermarq.hamiltonian import HamiltonianSimulation

circ_braket = HamiltonianSimulation(3)
circ_braket.run(1000)

circ_snuqs = HamiltonianSimulation(3, backend='snuqs')
circ_snuqs.run(1000)


from quantum_volume import QuantumVolume

circ_braket = QuantumVolume(3)
circ_braket.run(1000)

circ_snuqs = QuantumVolume(3, backend='snuqs')
circ_snuqs.run(1000)
"""