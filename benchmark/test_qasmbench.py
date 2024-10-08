from qasmbench import QASMBench

braket = QASMBench("adder_n4")
braket.run(1000)

snuqs = QASMBench("adder_n4", backend='snuqs')
snuqs.run(1000)